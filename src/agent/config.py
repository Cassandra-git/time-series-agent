"""Agent settings, read from environment variables.

No machine-specific address is stored in the code: each team member sets
OLLAMA_HOST for their setup (see README, section 6).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    # Context window. Larger = more RAM for the KV cache; 8192 fits the
    # system prompt, the tool schemas (~3k tokens) and a few turns.
    num_ctx: int = 8192
    # Upper bound on LLM <-> tool round trips for one user message.
    max_tool_rounds: int = 12
    # Where cleaned / converted Parquet files are written.
    processed_dir: str = "datasets/processed"
    # Where chart JSON files are written (read by tools/visualisation_tools.py).
    figures_dir: str = "outputs/figures"

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.getenv("OLLAMA_HOST", cls.ollama_host).strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = "http://" + host  # Ollama itself accepts "0.0.0.0:11434"
        return cls(
            ollama_host=host,
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", cls.num_ctx)),
            max_tool_rounds=int(os.getenv("TS_AGENT_MAX_TOOL_ROUNDS", cls.max_tool_rounds)),
            processed_dir=os.getenv("TS_AGENT_PROCESSED_DIR", cls.processed_dir),
            figures_dir=os.getenv("TS_AGENT_OUTPUT_DIR", cls.figures_dir),
        )
