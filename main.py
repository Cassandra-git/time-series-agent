import argparse
import json
import os
from pathlib import Path

from ollama import Client

from agent.builder import answer_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, help="Write the run and final answer to a JSON file.")
    parser.add_argument("--dataset", help="Path to the Parquet file to analyze.")
    parser.add_argument(
        "question", nargs="?", default="Donne moi la définition d'une analyse de données"
    )
    args = parser.parse_args()
    client = Client()
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    tool_calls = [] if args.trace else None
    if args.dataset:
        answer = answer_question(
            args.question, args.dataset, client=client, model=model, tool_trace=tool_calls
        )
    else:
        response = client.chat(
            model=model, messages=[{"role": "user", "content": args.question}]
        )
        answer = response.message.content

    if args.trace:
        trace = {
            "question": args.question,
            "dataset": args.dataset,
            "model": model,
            "tool_calls": tool_calls,
            "answer": answer,
        }
        args.trace.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(answer)


if __name__ == "__main__":
    main()