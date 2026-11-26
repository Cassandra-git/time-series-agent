"""Exercise persistence and recovery using deterministic models, without Ollama."""

import os
import subprocess
import sys

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent.conversation import (
    AgentSession,
    build_chat_model,
    build_conversation_graph,
    generate_conversation_title,
)
from agent.storage import open_storage


class ScriptedModel(BaseChatModel):
    replies: list[str]
    seen: list = Field(default_factory=list)
    fail_next: bool = False

    @property
    def _llm_type(self):
        return "scripted-memory-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated Ollama failure")
        response = AIMessage(content=self.replies.pop(0))
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_two_threads_receive_only_their_own_history(tmp_path):
    model = ScriptedModel(replies=["Bonjour P17", "Bonjour T42", "P17", "T42"])
    title_model = ScriptedModel(replies=["Suivi du capteur P17", "Suivi du capteur T42"])
    with open_storage(tmp_path / "memory.sqlite") as (store, saver):
        graph = build_conversation_graph(model, saver)
        first = AgentSession(graph, store, title_model, store.create().thread_id)
        second = AgentSession(graph, store, title_model, store.create().thread_id)
        first.send("Mon capteur est P17")
        second.send("Mon capteur est T42")
        assert first.send("Quel capteur ?").answer == "P17"
        assert second.send("Quel capteur ?").answer == "T42"
        assert [m.text for m in model.seen[2] if m.type == "human"] == [
            "Mon capteur est P17", "Quel capteur ?"
        ]
        assert [m.text for m in model.seen[3] if m.type == "human"] == [
            "Mon capteur est T42", "Quel capteur ?"
        ]
        assert len(title_model.seen) == 2
        assert len(first.history()) == 4
        assert store.get(first.thread_id).title == "Suivi du capteur P17"
        assert "Mon capteur est P17" in title_model.seen[0][-1].text
        assert "Bonjour P17" in title_model.seen[0][-1].text


def test_history_and_title_survive_a_new_process(tmp_path):
    path = tmp_path / "memory.sqlite"
    with open_storage(path) as (store, saver):
        conversation = store.create()
        session = AgentSession(
            build_conversation_graph(ScriptedModel(replies=["Bonjour P17"]), saver),
            store,
            ScriptedModel(replies=["Suivi du capteur P17"]),
            conversation.thread_id,
        )
        session.send("Mon capteur est P17")

    script = """
import sys
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from agent.storage import open_storage
from agent.conversation import AgentSession, build_conversation_graph
with open_storage(sys.argv[1]) as (store, saver):
    model = FakeMessagesListChatModel(responses=[AIMessage(content='Toujours P17')])
    graph = build_conversation_graph(model, saver)
    session = AgentSession(graph, store, model, sys.argv[2])
    assert [m.text for m in session.history()] == ['Mon capteur est P17', 'Bonjour P17']
    assert store.get(sys.argv[2]).title == 'Suivi du capteur P17'
    assert session.send('Quel capteur ?').answer == 'Toujours P17'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path), conversation.thread_id],
        check=True, env=os.environ.copy(), capture_output=True, text=True,
    )
    with open_storage(path) as (store, saver):
        graph = build_conversation_graph(ScriptedModel(replies=[]), saver)
        assert len(graph.get_state({"configurable": {"thread_id": conversation.thread_id}}).values["messages"]) == 4


def test_failed_turn_can_be_retried_after_reopening_without_duplicate_question(tmp_path):
    path = tmp_path / "memory.sqlite"
    with open_storage(path) as (store, saver):
        conversation = store.create("Titre manuel")
        model = ScriptedModel(replies=[], fail_next=True)
        session = AgentSession(build_conversation_graph(model, saver), store, model, conversation.thread_id)
        with pytest.raises(RuntimeError, match="Simulated"):
            session.send("Mon capteur est P17")
        assert session.has_pending_turn()
        with pytest.raises(ValueError, match="/retry"):
            session.send("Une autre question")

    with open_storage(path) as (store, saver):
        model = ScriptedModel(replies=["Bonjour P17"])
        session = AgentSession(build_conversation_graph(model, saver), store, model, conversation.thread_id)
        assert session.retry().answer == "Bonjour P17"
        assert [m.text for m in session.history() if m.type == "human"] == ["Mon capteur est P17"]
        assert not session.has_pending_turn()
        assert store.get(conversation.thread_id).title == "Titre manuel"
        with pytest.raises(ValueError, match="Aucun échange"):
            session.retry()


def test_title_failure_preserves_answer_and_can_be_retried_on_next_turn(tmp_path):
    model = ScriptedModel(replies=["Bonjour P17", "P17"])
    title_model = ScriptedModel(replies=["Suivi du capteur P17"], fail_next=True)
    with open_storage(tmp_path / "memory.sqlite") as (store, saver):
        session = AgentSession(
            build_conversation_graph(model, saver), store, title_model, store.create().thread_id
        )
        result = session.send("Mon capteur est P17")
        assert result.answer == "Bonjour P17"
        assert result.title_warning
        assert len(session.history()) == 2
        result = session.send("Quel capteur ?")
        assert result.title == "Suivi du capteur P17"
        assert result.title_warning is None
        assert len(session.history()) == 4


def test_empty_message_and_unknown_thread_are_rejected(tmp_path):
    model = ScriptedModel(replies=[])
    with open_storage(tmp_path / "memory.sqlite") as (store, saver):
        graph = build_conversation_graph(model, saver)
        with pytest.raises(KeyError):
            AgentSession(graph, store, model, "missing")
        session = AgentSession(graph, store, model, store.create().thread_id)
        with pytest.raises(ValueError, match="texte"):
            session.send("  ")
        assert session.history() == []


def test_title_cleanup_and_empty_output():
    assert generate_conversation_title(ScriptedModel(replies=['"Suivi du capteur P17"']), []) == "Suivi du capteur P17"
    with pytest.raises(ValueError, match="titre vide"):
        generate_conversation_title(ScriptedModel(replies=[" "]), [])


def test_model_configuration_uses_plain_text_and_qwen():
    model = build_chat_model("http://localhost:11434/")
    assert model.model == "qwen3:8b"
    assert model.base_url == "http://localhost:11434"
    assert model.reasoning is False
    assert model.format != "json"
    with pytest.raises(ValueError, match="URL"):
        build_chat_model("localhost:11434")


def test_terminal_commands_create_and_reopen_a_conversation(tmp_path, monkeypatch, capsys):
    from agent import cli

    path = tmp_path / "terminal.sqlite"
    monkeypatch.setattr(sys, "argv", ["agent.cli", "--database", str(path)])
    model = ScriptedModel(replies=["Bonjour P17", "Suivi du capteur P17"])
    monkeypatch.setattr(cli, "build_chat_model", lambda *args: model)
    commands = iter(["/list", "/new", "Mon capteur est P17", "/history", "/list", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    cli.main()
    output = capsys.readouterr().out
    assert "Aucune discussion" in output
    assert "Qwen : Bonjour P17" in output
    assert "[human] Mon capteur est P17" in output
    assert "Suivi du capteur P17" in output

    with open_storage(path) as (store, saver):
        thread_id = store.list()[0].thread_id
    model = ScriptedModel(replies=["P17"])
    monkeypatch.setattr(cli, "build_chat_model", lambda *args: model)
    commands = iter([f"/open {thread_id}", "Quel capteur ?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    cli.main()
    assert "Qwen : P17" in capsys.readouterr().out
    assert [m.text for m in model.seen[0] if m.type == "human"] == [
        "Mon capteur est P17", "Quel capteur ?"
    ]
