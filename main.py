"""Command-line interface of the time series agent.

    uv run python main.py --dataset datasets/sample/ASE4.csv
    uv run python main.py --dataset datasets/sample/ASE4.csv "Quelle est la moyenne de ASE_T ?"

Without a question, starts a multi-turn conversation (type "exit" to quit).
"""

import argparse
import json
from pathlib import Path

from agent.builder import AgentSession, Turn, build_agent
from agent.config import Settings
from agent.graph import answer_interrupt_in_terminal


def run_turn(session: AgentSession, question: str, trace: list | None) -> Turn:
    turn = session.send(question)
    calls = list(turn.tool_calls)
    figures = list(turn.figures)
    while turn.question:  # the cleaning workflow asks the user in the terminal
        turn = session.resume(answer_interrupt_in_terminal(turn.question))
        calls += turn.tool_calls
        figures += turn.figures
    for call in calls:
        print(f"  [outil] {call['tool']}({json.dumps(call['arguments'], ensure_ascii=False)})")
    print(f"\n{turn.answer}")
    for figure in figures:
        print(f"  [graphique] {figure}")
    if trace is not None:
        trace.append({"question": question, "tool_calls": calls, "figures": figures, "answer": turn.answer})
    return turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Time series analysis agent (Qwen + Ollama).")
    parser.add_argument("--dataset", help="CSV, Excel or Parquet file to analyse.")
    parser.add_argument("--trace", type=Path, help="Write questions, tool calls and answers to a JSON file.")
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    args = parser.parse_args()

    settings = Settings.from_env()
    session = AgentSession(build_agent(settings), settings)
    if args.dataset:
        session.load_dataset(args.dataset)
    print(f"Modèle : {settings.ollama_model} @ {settings.ollama_host}")

    trace = [] if args.trace else None
    try:
        if args.question:
            run_turn(session, args.question, trace)
        else:
            print("Posez vos questions (exit pour quitter).")
            while True:
                try:
                    question = input("\n> ").strip()
                except EOFError:
                    break
                if question.lower() in {"exit", "quit", "q"}:
                    break
                if question:
                    run_turn(session, question, trace)
    finally:
        if args.trace:
            args.trace.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
