"""Assemble the time series agent: Qwen (Ollama) + tools + LangGraph ReAct loop.

    session = AgentSession(build_agent())
    session.load_dataset("datasets/sample/ASE4.csv")
    turn = session.send("Trace ASE_NO2 sur toute la période")
    while turn.question:                    # cleaning workflow asks the user
        turn = session.resume(answer_to(turn.question))
    print(turn.answer, turn.figures)

The same AgentSession is meant to back the CLI (main.py) and, later, the
Streamlit interface: one session = one LangGraph thread = one conversation.
"""

import json
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from typing_extensions import NotRequired

from agent.config import Settings
from agent.prompts import SYSTEM_PROMPT, dataset_context
from tools.agent_tools import dataset_state, make_agent_tools

# Errors a tool may raise because of bad arguments: they are sent back to the
# model as a ToolMessage so it can correct itself. Other errors stop the run.
RECOVERABLE_TOOL_ERRORS = (ValueError, KeyError, TypeError, FileNotFoundError)


class TimeSeriesAgentState(AgentState):
    """Conversation state: messages plus the dataset the tools work on."""
    source_path: NotRequired[str]   # file given by the user (CSV, Excel, Parquet)
    dataset_path: NotRequired[str]  # Parquet file read by the analysis tools
    dataset_name: NotRequired[str]
    cleaned: NotRequired[bool]


def build_model(settings: Settings) -> ChatOllama:
    """Qwen through Ollama, configured for tool calling."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        temperature=0,
        reasoning=False,  # qwen3 "thinking" is slow on CPU and not needed for tool choice
        num_ctx=settings.num_ctx,
    )


def _prompt(state: dict) -> list:
    system = SystemMessage(f"{SYSTEM_PROMPT}\n{dataset_context(state)}")
    return [system, *state["messages"]]


def build_agent(settings: Settings | None = None, model: BaseChatModel | None = None, checkpointer=None):
    """Compile the agent graph. `model` can be replaced in tests."""
    settings = settings or Settings.from_env()
    tools = ToolNode(make_agent_tools(settings.processed_dir), handle_tool_errors=RECOVERABLE_TOOL_ERRORS)
    return create_react_agent(
        model or build_model(settings),
        tools,
        prompt=_prompt,
        state_schema=TimeSeriesAgentState,
        checkpointer=checkpointer or InMemorySaver(),
        name="time_series_agent",
    )


@dataclass
class Turn:
    """What the interface needs to show after one call to the agent."""
    answer: str = ""
    # Set when the agent is paused on a question for the user (cleaning workflow).
    question: dict | None = None
    figures: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


class AgentSession:
    """One conversation with the agent (multi-turn memory via the checkpointer)."""

    def __init__(self, agent, settings: Settings | None = None, thread_id: str | None = None):
        self.agent = agent
        self.settings = settings or Settings.from_env()
        self.config = {
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
            # Each tool round is two graph steps (model + tools).
            "recursion_limit": 2 * self.settings.max_tool_rounds + 1,
        }
        self._pending: dict = {}

    @property
    def state(self) -> dict:
        return self.agent.get_state(self.config).values

    def load_dataset(self, file_path: str) -> dict:
        """Select the dataset from the interface (file chooser, upload, CLI flag)."""
        update = dataset_state(file_path, self.settings.processed_dir)
        if self.state:
            self.agent.update_state(self.config, update)
        else:
            self._pending.update(update)  # sent with the first message
        return update

    def send(self, text: str) -> Turn:
        payload = {**self._pending, "messages": [("user", text)]}
        self._pending = {}
        return self._run(payload)

    def resume(self, answer) -> Turn:
        """Answer the question the agent is paused on."""
        return self._run(Command(resume=answer))

    def _run(self, payload) -> Turn:
        before = len(self.state.get("messages", []))
        result = self.agent.invoke(payload, self.config)
        turn = Turn()
        for message in result["messages"][before:]:
            for call in getattr(message, "tool_calls", None) or []:
                turn.tool_calls.append({"tool": call["name"], "arguments": call["args"]})
            if isinstance(message, ToolMessage):
                figure = _figure_path(message.content)
                if figure:
                    turn.figures.append(figure)
        if result.get("__interrupt__"):
            turn.question = result["__interrupt__"][0].value
        else:
            turn.answer = result["messages"][-1].content
        return turn


def _figure_path(content) -> str | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    return data.get("figure_path") if isinstance(data, dict) else None
