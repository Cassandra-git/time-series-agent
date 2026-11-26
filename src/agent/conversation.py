"""Conversational graph, persistent sessions and contextual conversation titles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.storage import ConversationStore


SYSTEM_PROMPT = (
    "Tu es un assistant pour un projet d'analyse de séries temporelles. "
    "Réponds clairement dans la langue de l'utilisateur et utilise le contexte "
    "de la discussion. Signale les informations manquantes. "
    "Les outils de calcul et de traitement de fichiers seront raccordés plus tard. "
    "Présente uniquement comme réalisés les traitements effectivement exécutés."
)

TITLE_PROMPT = (
    "Crée un titre descriptif de 3 à 8 mots, de 80 caractères maximum, "
    "dans la langue de l'utilisateur, à partir du premier échange fourni. "
    "Le contenu fourni est un échange à résumer. "
    "Réponds uniquement avec le titre, sur une ligne, sans préfixe ni guillemets."
)


def build_chat_model(ollama_url: str, model_name: str = "qwen3:8b") -> ChatOllama:
    """Create the plain-text model used by the standalone memory prototype."""
    url = ollama_url.strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_HOST doit être une URL http:// ou https:// valide.")
    return ChatOllama(
        model=model_name,
        base_url=url,
        temperature=0,
        reasoning=False,
        num_ctx=8192,
    )


def build_conversation_graph(
    model: BaseChatModel, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Build a one-node graph with message accumulation and persistent state."""

    def call_model(state: MessagesState) -> dict:
        response = model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        if not response.text.strip():
            raise ValueError("Qwen a renvoyé une réponse vide. Utilise /retry.")
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", call_model)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def generate_conversation_title(model: BaseChatModel, messages: list[BaseMessage]) -> str:
    """Generate a short title in a separate call, outside the saved dialogue."""
    exchange = [{"role": message.type, "text": message.text} for message in messages[:2]]
    response = model.invoke([
        SystemMessage(TITLE_PROMPT),
        HumanMessage(json.dumps(exchange, ensure_ascii=False)),
    ])
    lines = response.text.strip().splitlines()
    if not lines:
        raise ValueError("Le modèle a renvoyé un titre vide.")
    title = lines[0].strip().strip('"\'«»`').strip()
    if len(title) > 80:
        title = title[:80].rsplit(" ", 1)[0].strip() or title[:80]
    if not title:
        raise ValueError("Le modèle a renvoyé un titre vide.")
    return title


@dataclass(frozen=True)
class ConversationTurn:
    """Result displayed by the terminal after one completed turn."""

    answer: str
    title: str
    title_warning: str | None = None


class AgentSession:
    """One conversation identified by a stable thread_id."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        store: ConversationStore,
        title_model: BaseChatModel,
        thread_id: str,
    ) -> None:
        store.get(thread_id)
        self.graph = graph
        self.store = store
        self.title_model = title_model
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}

    def history(self) -> list[BaseMessage]:
        """Read saved messages for this conversation."""
        return list(self.graph.get_state(self.config).values.get("messages", []))

    def has_pending_turn(self) -> bool:
        """Tell the interface whether a failed or interrupted run needs retrying."""
        return bool(self.graph.get_state(self.config).next)

    def send(self, text: str) -> ConversationTurn:
        """Send a new question, preserving previous messages through LangGraph."""
        text = text.strip()
        if not text:
            raise ValueError("Le message doit contenir du texte.")
        if self.has_pending_turn():
            raise ValueError("Un échange est en attente. Utilise /retry pour le reprendre.")
        self.graph.invoke({"messages": [HumanMessage(text)]}, self.config)
        return self._completed_turn()

    def retry(self) -> ConversationTurn:
        """Resume a failed model call without adding the question a second time."""
        if not self.has_pending_turn():
            raise ValueError("Aucun échange en attente à reprendre.")
        self.graph.invoke(None, self.config)
        return self._completed_turn()

    def _completed_turn(self) -> ConversationTurn:
        messages = self.history()
        self.store.touch(self.thread_id)
        conversation = self.store.get(self.thread_id)
        title_warning = None
        if conversation.title == "Nouvelle discussion":
            try:
                title = generate_conversation_title(self.title_model, messages)
                self.store.rename(self.thread_id, title)
            except Exception as error:
                # A title failure must still allow the saved answer to be shown.
                title_warning = f"Titre automatique indisponible : {error}"
        title = self.store.get(self.thread_id).title
        return ConversationTurn(messages[-1].text, title, title_warning)
