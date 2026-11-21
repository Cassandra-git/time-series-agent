"""Agent integration tests, without Ollama.

A scripted chat model plays the LLM: it emits the tool calls we choose, so the
tests check the wiring (state, tools, errors, interrupts, memory), not Qwen.
"""

import json
from pathlib import Path

import polars as pl
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent.builder import AgentSession, build_agent
from agent.config import Settings


class ScriptedModel(FakeMessagesListChatModel):
    """Returns the scripted messages in order and records what it was sent."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def call(name: str, call_id: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def tool_results(session: AgentSession) -> dict[str, dict]:
    return {
        m.tool_call_id: json.loads(m.content)
        for m in session.state["messages"]
        if isinstance(m, ToolMessage) and m.content.startswith("{")
    }


@pytest.fixture
def csv_path(tmp_path) -> Path:
    path = tmp_path / "sensor.csv"
    pl.DataFrame({
        "timestamp": [f"2026-01-01 0{h}:00:00" for h in range(6)],
        "temperature": [10.0, 12.0, None, 14.0, 16.0, 18.0],
        "pressure": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    }).write_csv(path)
    return path


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TS_AGENT_OUTPUT_DIR", str(tmp_path / "figures"))
    settings = Settings(processed_dir=str(tmp_path / "processed"))

    def make(*responses) -> tuple[AgentSession, ScriptedModel]:
        model = ScriptedModel(responses=list(responses), seen=[])
        return AgentSession(build_agent(settings, model=model), settings), model

    return make


def test_statistics_question_uses_the_loaded_dataset(make_session, csv_path):
    session, _ = make_session(
        call("inspect_dataset", "c1"),
        call("describe_column", "c2", column="temperature"),
        AIMessage(content="La température moyenne est 14."),
    )
    session.load_dataset(str(csv_path))
    turn = session.send("Moyenne de la température ?")

    assert [c["tool"] for c in turn.tool_calls] == ["inspect_dataset", "describe_column"]
    results = tool_results(session)
    assert [c["name"] for c in results["c1"]["columns"]] == ["timestamp", "temperature", "pressure"]
    assert results["c2"]["mean"] == 14.0
    assert results["c2"]["null_count"] == 1
    assert turn.answer == "La température moyenne est 14."
    assert csv_path.read_text().startswith("timestamp")  # source untouched


def test_chart_is_saved_and_reported(make_session, csv_path):
    session, _ = make_session(
        call("plot_time_series", "c1", value_cols=["temperature", "pressure"], time_col="timestamp"),
        AIMessage(content="Voici la courbe."),
    )
    session.load_dataset(str(csv_path))
    turn = session.send("Trace les capteurs")

    assert len(turn.figures) == 1
    assert Path(turn.figures[0]).is_file()


def test_tool_error_is_sent_back_to_the_model(make_session, csv_path):
    session, model = make_session(
        call("describe_column", "c1", column="humidity"),
        AIMessage(content="Cette colonne n'existe pas."),
    )
    session.load_dataset(str(csv_path))
    turn = session.send("Statistiques de l'humidité")

    error = next(m for m in session.state["messages"] if isinstance(m, ToolMessage))
    assert error.status == "error"
    assert "humidity" in error.content
    assert turn.answer == "Cette colonne n'existe pas."


def test_without_dataset_the_tool_explains_what_is_missing(make_session):
    session, _ = make_session(call("inspect_dataset", "c1"), AIMessage(content="Chargez un fichier."))
    session.send("Quelles colonnes ?")
    error = next(m for m in session.state["messages"] if isinstance(m, ToolMessage))
    assert "No dataset is loaded" in error.content


def test_load_dataset_tool_changes_the_active_dataset(make_session, csv_path):
    session, model = make_session(
        call("load_dataset", "c1", file_path=str(csv_path)),
        AIMessage(content="Fichier chargé."),
    )
    session.send(f"Charge {csv_path}")
    state = session.state
    assert state["dataset_name"] == "sensor.csv"
    assert state["dataset_path"].endswith(".parquet")
    assert state["cleaned"] is False


def test_conversation_keeps_history_and_dataset_between_turns(make_session, csv_path):
    session, model = make_session(
        AIMessage(content="Bonjour."),
        call("describe_column", "c1", column="pressure"),
        AIMessage(content="Moyenne 1.25."),
    )
    session.load_dataset(str(csv_path))
    session.send("Salut")
    turn = session.send("Et la pression ?")

    assert turn.answer == "Moyenne 1.25."
    last_prompt = model.seen[-1]
    assert "sensor.csv" in last_prompt[0].content  # dataset context in the system prompt
    assert [m.content for m in last_prompt if m.type == "human"] == ["Salut", "Et la pression ?"]


def test_cleaning_questions_reach_the_user_and_the_cleaned_file_becomes_active(make_session, csv_path):
    session, _ = make_session(
        call("clean_dataset", "c1"),
        AIMessage(content="Données nettoyées."),
    )
    session.load_dataset(str(csv_path))
    raw_path = session.state["dataset_path"] if session.state else None
    source = csv_path.read_bytes()

    turn = session.send("Nettoie les données")
    asked = []
    while turn.question:
        kind = turn.question["type"]
        asked.append(kind)
        if kind == "completeness_requirement":
            answer = "yes"
        elif kind == "cleaning_clarification":
            answer = {"question_1": "mesure", "question_2": "inconnus", "question_3": "temperature=linear"}
        elif kind == "cleaning_approval":
            answer = {
                "approved_ids": [r["id"] for r in turn.question["recommendations"]
                                 if r["technical_status"] == "valid"
                                 and r.get("parameters", {}).get("method") != "keep"],
                "extra_recommendations": [],
            }
        else:
            pytest.fail(f"Unexpected question: {turn.question}")
        turn = session.resume(answer)

    assert asked[0] == "completeness_requirement"
    assert asked[-1] == "cleaning_approval"
    assert turn.answer == "Données nettoyées."
    state = session.state
    assert state["cleaned"] is True
    assert state["dataset_path"] != raw_path
    cleaned = pl.read_parquet(state["dataset_path"])
    assert cleaned["temperature"].null_count() == 0
    assert csv_path.read_bytes() == source
