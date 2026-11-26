"""Standalone terminal for the conversation-memory feature.

Run from the repository root: uv run python -m agent.cli
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent.conversation import AgentSession, build_chat_model, build_conversation_graph
from agent.storage import open_storage


HELP = """Commandes disponibles
  /new            Créer une discussion
  /list           Lister les discussions et leurs identifiants
  /open ID        Ouvrir une discussion avec son identifiant complet
  /history        Afficher les messages de la discussion active
  /retry          Reprendre une réponse interrompue ou échouée
  /help           Afficher cette aide
  /quit           Quitter

Écris une question pour discuter avec Qwen.
"""


def main() -> None:
    """Open storage once and keep the graph alive throughout the terminal loop."""
    parser = argparse.ArgumentParser(description="Conversations persistantes avec Qwen.")
    parser.add_argument(
        "--database", type=Path, default=Path("runtime/conversations.sqlite")
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    args = parser.parse_args()
    model = build_chat_model(args.ollama_host, args.model)

    with open_storage(args.database) as (store, checkpointer):
        graph = build_conversation_graph(model, checkpointer)
        session = None
        print(f"Modèle : {args.model} | Ollama : {args.ollama_host}")
        print(f"Base de données : {args.database.expanduser().resolve()}")
        print(HELP)
        while True:
            try:
                text = input("\n> ").strip()
                if not text:
                    continue
                command, _, argument = text.partition(" ")
                argument = argument.strip()
                if command == "/quit":
                    break
                if command == "/help":
                    print(HELP)
                    continue
                if command == "/new":
                    conversation = store.create()
                    session = AgentSession(graph, store, model, conversation.thread_id)
                    print(f"Discussion créée : {conversation.thread_id}")
                    continue
                if command == "/list":
                    conversations = store.list()
                    if not conversations:
                        print("Aucune discussion. Utilise /new pour commencer.")
                    for conversation in conversations:
                        print(f"{conversation.thread_id} | {conversation.title}")
                    continue
                if command == "/open":
                    conversation = store.get(argument)
                    session = AgentSession(graph, store, model, conversation.thread_id)
                    print(f"Discussion ouverte : {conversation.title} ({conversation.thread_id})")
                    print(f"{len(session.history())} messages sauvegardés.")
                    if session.has_pending_turn():
                        print("Une réponse est en attente. Utilise /retry.")
                    continue
                if session is None:
                    print("Sélectionne une discussion avec /new ou /open ID.")
                    continue
                if command == "/history":
                    for message in session.history():
                        print(f"[{message.type}] {message.text}")
                    continue
                if command == "/retry":
                    turn = session.retry()
                elif command.startswith("/"):
                    print("Commande inconnue. Utilise /help.")
                    continue
                else:
                    print("Qwen prépare sa réponse…", flush=True)
                    turn = session.send(text)
                print(f"\nQwen : {turn.answer}")
                print(f"Discussion : {turn.title}")
                if turn.title_warning:
                    print(turn.title_warning)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nFermeture. Les checkpoints déjà écrits sont conservés.")
                break
            except Exception as error:
                # Keep the selected conversation available after a recoverable failure.
                print(f"Erreur : {error}")
                print("Utilise /retry si une réponse est en attente, ou /help.")


if __name__ == "__main__":
    main()