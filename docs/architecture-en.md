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

### `src/ts_agent/`

Main Python package containing the application's business logic and agent implementation.

- `__init__.py`: identifies the directory as a Python package.
- `config.py`: centralizes configuration, including Ollama host, model name, and application settings.

### `src/ts_agent/agent/`

Contains agent construction and orchestration.

- `__init__.py`: Python package initialization.
- `builder.py`: initializes the LLM, registers tools, and constructs the agent.
- `graph.py`: defines explicit LangGraph workflows and state management when required.
- `prompts.py`: stores system prompts and agent instructions.

### `src/ts_agent/tools/`

Contains the interfaces exposed to the LLM.

- `__init__.py`: Python package initialization.
- `cleaning_tools.py`: tools for loading and inspecting datasets.
- `analysis_tools.py`: tools for statistics, anomaly detection, and forecasting.
- `visualisation_tools.py`: tools for requesting visualizations.

Tools validate inputs, call the relevant services, and return structured results.

### `src/ts_agent/services/`

Contains reusable Python functions independent of the LLM framework.

- `__init__.py`: Python package initialization.
- `data_loader.py`: loads and validates CSV datasets.
- `preprocessing.py`: cleans data, handles missing values, and prepares timestamps.
- `statistics.py`: computes descriptive statistics.
- `anomalies.py`: implements anomaly detection algorithms.
- `forecasting.py`: implements time series forecasting methods.
- `plotting.py`: generates charts using Plotly.

### `app/`

Contains the user interface.

- `streamlit_app.py`: manages file uploads, user questions, conversation display, and result visualization.

### `data/`

Contains datasets used for development and demonstration.

- `sample/`: contains csv files of data

Large, generated, or private datasets should not be committed.

### `tests/`

Contains automated tests.

- `test_services.py`: validates numerical and data-processing functions.
- `test_tools.py`: verifies tool inputs and outputs.
- `test_agent.py`: tests agent integration and tool selection.

### `docs/`

Contains project documentation.

- `architecture.md`: architecture documentation in English.
- `architecture.fr.md`: architecture documentation in French.

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

## 5. Implementation Note

`builder.py` can use LangChain's agent implementation, which is built on LangGraph. A separate `graph.py` is only necessary when custom workflow states, transitions, or orchestration are required.

All listed modules describe the target architecture and may be implemented progressively.
