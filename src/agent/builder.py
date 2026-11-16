
from langchain_ollama import ChatOllama


def build_model(ollama_url: str) -> ChatOllama:
    """
    Create the Qwen model using the Ollama URL
    supplied by the user at runtime.

    No machine-specific URL is stored in the source code.
    """

    url = ollama_url.strip().rstrip("/")

    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "The Ollama URL must start with http:// or https://."
        )

    return ChatOllama(
        model="qwen3:8b",
        base_url=ollama_url.strip(),
        temperature=0,
        format="json",
        think=False,
        num_predict=2048,
    )