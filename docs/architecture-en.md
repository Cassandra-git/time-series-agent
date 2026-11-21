# Project Architecture

## 1. Overview

The project follows a modular architecture separating the conversational agent, LLM-accessible tools, data processing services, and user interface.

**Processing flow:**

```text
User
  |
  v
Streamlit Interface
  |
  v
LangChain / LangGraph Agent
  |
  +---- Ollama / Qwen (language understanding)
  |
  v
LangChain Tools
  |
  v
Python Services
  |
  v
Analysis Results
  |
  v
Agent Response + Visualizations
```

The LLM interprets requests and decides which tools to invoke. Python services perform the actual data processing and numerical computations.

## 2. Directory Structure

### `.github/workflows/`

Contains GitHub Actions automation.

- `tests.yml`: runs automated checks when code is pushed or a Pull Request is opened.

This directory is unrelated to LangGraph workflows.

### `.venv/`

Local Python virtual environment containing the interpreter and installed dependencies.

It must be excluded from Git.

### `src/`

Main Python code. `pyproject.toml` installs its three packages (`agent`, `tools`, `services`): import them directly (`from services.statistics import ...`), without a `src.` prefix.

### `src/agent/`

Contains agent construction and orchestration.

- `config.py`: settings read from the environment (`OLLAMA_HOST`, `OLLAMA_MODEL`, output directories, maximum tool rounds).
- `builder.py`: initializes the LLM (`ChatOllama`), registers the tools, builds the LangGraph ReAct agent (`create_react_agent`) with a state and conversation memory, and provides `AgentSession`, used by the CLI and later by Streamlit.
- `graph.py`: guided LangGraph cleaning workflow (audit, business questions, human approval, Parquet export). It is started by the `clean_dataset` tool and can also run on its own in the terminal.
- `prompts.py`: system prompt and active-dataset context.

### `src/tools/`

Contains the interfaces exposed to the LLM.

- `agent_tools.py`: the agent's tool catalogue. The tools read the active dataset from the agent state, so the LLM never has to type a file path.
- `cleaning_tools.py`: dataset inspection and cleaning tools.
- `analysis_tools.py`: statistics tools.
- `visualisation_tools.py`: chart tools.

Tools validate inputs, call the relevant services, and return structured results.

### `src/services/`

Contains reusable Python functions independent of the LLM framework.

- `data_loader.py`: loads, validates and converts CSV, Excel and Parquet datasets to Parquet.
- `preprocessing.py`: cleans data, handles missing values, and prepares timestamps.
- `statistics.py`: computes descriptive statistics.
- `anomalies.py`: anomaly detection algorithms (planned).
- `forecasting.py`: forecasting methods (planned).
- `plotting.py`: generates charts using Plotly.

### `app/`

Contains the user interface.

- `streamlit_app.py`: manages file uploads, user questions, conversation display, and result visualization.

### `datasets/`

Contains datasets used for development and demonstration.

- `sample/`: CSV sample files.
- `processed/`: Parquet files converted or cleaned by the agent (not committed).

Large, generated, or private datasets should not be committed.

### `tests/`

Contains automated tests.

- `test_services.py`: validates numerical and data-processing functions.
- `test_tools.py`: verifies tool inputs and outputs.
- `test_agent.py`: tests agent integration and tool selection.

### `docs/`

Contains project documentation.

- `architecture-en.md`: architecture documentation in English.
- `architecture-fr.md`: architecture documentation in French.

### `experiments/`

Contains exploratory notebooks and benchmarks.

- `model_comparison.ipynb`: compares LLMs, prompts, or analysis approaches.

Experimental code should not become a production dependency.

## 3. Root Files

| File | Purpose |
|---|---|
| `README.md` | Main project documentation in English |
| `pyproject.toml` | Python package metadata, dependencies, and tool configuration |
| `.env.example` | Example environment variables without private values |
| `.gitignore` | Files and directories excluded from Git |
| `Dockerfile` | Instructions for building the Python application container |
| `.dockerignore` | Files excluded from the Docker build context |

## 4. Design Principles

**Separation of concerns:** the agent orchestrates, tools expose capabilities, and services perform computations.

**Modularity:** each component can be developed and tested independently.

**Reusability:** services remain usable without LangChain or Ollama.

**Configurability:** local Ollama addresses and model choices are not hardcoded.

**Collaboration:** feature branches, Pull Requests, and automated tests support parallel development.

**Incremental development:** start with CSV loading and basic statistics before introducing advanced workflows, forecasting, and Docker.

## 5. How the Agent Works

`builder.py` uses LangGraph's ReAct agent: the LLM picks a tool, LangGraph runs it and returns the result to the LLM, until the LLM answers without calling a tool.

- **State:** besides the messages, the state holds the source file and the active Parquet file (`dataset_path`). `load_dataset` and `clean_dataset` update it; the other tools read it.
- **Memory:** a checkpointer keeps the state per `thread_id`. One `AgentSession` is one multi-turn conversation.
- **Human approval:** `clean_dataset` runs the `graph.py` workflow. Each of its questions goes through `interrupt()`: the agent pauses, the interface shows the question, then `AgentSession.resume(answer)` resumes it.
- **Errors:** a tool argument error (unknown column, invalid date, etc.) is sent back to the LLM so it can fix its call.
- **Tests:** `tests/test_agent.py` replaces Qwen with a scripted model, so the assembly is tested without Ollama.

Anomaly detection and forecasting modules will be added progressively.
