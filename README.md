# Time Series AI Agent

A conversational AI agent for time series analysis, anomaly detection, visualization, and forecasting using natural language.

## 1. Project Overview

**Time Series AI Agent** is an academic project developed by a team of three students over eight weeks.

The objective is to build an intelligent assistant that allows users to upload time series datasets, ask questions in natural language, and receive statistical analyses, interactive visualizations, anomaly detection results, and forecasts.

The application combines a locally hosted large language model (LLM) with specialized Python tools.

The LLM interprets user requests and selects the appropriate tools, while data processing and numerical calculations are performed by dedicated Python libraries.

### Main Objectives

- Import and explore time series datasets from CSV files.
- Automatically identify datetime columns and numerical variables.
- Clean and preprocess time series data.
- Calculate descriptive statistics.
- Analyze trends and seasonality.
- Detect anomalies and unusual observations.
- Generate interactive visualizations.
- Forecast future values using statistical and machine learning models.
- Explain analytical results in natural language.
- Provide an interactive conversational interface.

## 2. Features

### Planned Features

- CSV file upload and validation.
- Automatic time series preprocessing.
- Descriptive statistical analysis.
- Trend and seasonality analysis.
- Anomaly detection.
- Interactive Plotly charts.
- Time series forecasting.
- Natural language explanations.
- Conversational agent powered by LangChain and LangGraph.
- Streamlit web interface.

### Currently Validated

- Local Qwen3:8b model running through Ollama.
- Python-to-Ollama communication from Ubuntu WSL.
- Docker Desktop integration with WSL.
- Communication between a Docker container and Ollama running on Windows.

The complete analytical agent and Docker application image are still under development.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Main programming language |
| Qwen3:8b | Local large language model |
| Ollama | Local LLM inference server |
| LangChain | LLM integration and tool definitions |
| LangGraph | Agent workflow orchestration |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Statsmodels | Statistical analysis and forecasting |
| Scikit-learn | Machine learning and anomaly detection |
| Plotly | Interactive visualizations |
| Streamlit | Web-based user interface |
| Pytest | Automated testing |
| Docker | Application containerization |
| GitHub Actions | Continuous integration |

**Note:** Dependency compatibility with Python 3.14 must be verified during installation.

## 4. System Architecture

The application follows a modular architecture that separates the user interface, conversational agent, tools, and analytical services.

```text
                       User
                        |
                        v
                 Streamlit Interface
                        |
                        v
                   LangGraph Agent
                        |
              +---------+---------+
              |                   |
              v                   v
         Qwen3:8b LLM       LangChain Tools
          via Ollama              |
                                  v
                            Python Services
                                  |
                +-----------------+-----------------+
                |                 |                 |
                v                 v                 v
           Data Loading      Statistics        Forecasting
           Preprocessing     Anomalies         Visualization
```

The language model is responsible for understanding requests and coordinating operations.

Data processing, statistical calculations, anomaly detection, and forecasting are handled by dedicated Python services rather than relying on the LLM to perform numerical computations.

## 5. Project Structure

```text
time-series-agent/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   └── ts_agent/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   ├── graph.py
│       │   └── prompts.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── data_tools.py
│       │   ├── analysis_tools.py
│       │   └── plot_tools.py
│       │
│       └── services/
│           ├── __init__.py
│           ├── data_loader.py
│           ├── preprocessing.py
│           ├── statistics.py
│           ├── anomalies.py
│           ├── forecasting.py
│           └── plotting.py
│
├── data/
│   └── sample/
│       └── sensor_data.csv
│
├── models/
│
├── tests/
│   ├── test_services.py
│   ├── test_tools.py
│   └── test_agent.py
│
├── docs/
│   ├── architecture.md
│   └── architecture.fr.md
│
├── experiments/
│   └── model_comparison.ipynb
│
├── main.py
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── README.fr.md
```

### Directory Responsibilities

| Directory | Description |
|---|---|
| `app/` | Streamlit user interface |
| `src/ts_agent/agent/` | LangGraph agent and conversation workflow |
| `src/ts_agent/tools/` | LangChain tools available to the agent |
| `src/ts_agent/services/` | Independent data processing and analytical functions |
| `data/sample/` | Example datasets |
| `models/` | Saved forecasting models, when required |
| `tests/` | Unit and integration tests |
| `docs/` | Technical documentation |
| `experiments/` | Model evaluation and experimentation |

The `models/` directory is intended for trained forecasting models. Qwen3:8b is managed by Ollama and does not need to be stored in this directory.

The root-level `main.py` is an initial script used to validate communication with Ollama.

## 6. Development Environments

The project supports three development configurations.

| Configuration | Python Environment | Ollama Server | API Address |
|---|---|---|---|
| Windows only | Windows PowerShell | Windows | `http://localhost:11434` |
| Windows + WSL 2 | Ubuntu WSL | Windows | `http://<WSL_GATEWAY_IP>:11434` |
| Docker Desktop | Docker container | Windows | `http://host.docker.internal:11434` |

The Python application and Ollama server do not need to run on the same operating system, but the API address must match the selected environment.

## 7. Installation — Windows Only

This configuration is intended for developers who run both Python and Ollama directly on Windows, without WSL.

### 7.1. Prerequisites

Install:

- Python 3.14: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads/win
- Ollama: https://ollama.com/
- Visual Studio Code (recommended): https://code.visualstudio.com/

Ensure that Python is accessible from the command line.

### 7.2. Clone the Repository

Open Windows PowerShell or the PowerShell terminal in VS Code.

```powershell
git clone <REPOSITORY_URL>
cd time-series-agent
```

### 7.3. Create a Virtual Environment

Check the Python version:

```powershell
py -3.14 --version
```

Create the environment:

```powershell
py -3.14 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, temporarily allow activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

Verify:

```powershell
python --version
```

Expected output:

```text
Python 3.14.x
```

### 7.4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

Dependencies must be declared in `pyproject.toml`.

### 7.5. Install and Configure Ollama

Verify the installation:

```powershell
ollama --version
```

Download Qwen3:8b:

```powershell
ollama pull qwen3:8b
```

Verify the model:

```powershell
ollama list
```

Ollama usually starts automatically on Windows and exposes its API at:

```text
http://localhost:11434
```

If the server is not running:

```powershell
ollama serve
```

Do not start another instance if Ollama is already using port `11434`.

### 7.6. Configure Environment Variables

```powershell
$env:OLLAMA_HOST="http://localhost:11434"
$env:OLLAMA_MODEL="qwen3:8b"
```

These variables apply to the current PowerShell session.

### 7.7. Test the Connection

```powershell
curl.exe http://localhost:11434/api/tags
```

A successful response returns JSON containing the available models.

Test Qwen directly:

```powershell
ollama run qwen3:8b
```

### 7.8. Run the Application

Ensure the virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python main.py
```

The application should connect to Ollama, accept a question, and display the model's response.

## 8. Installation — Windows + Ubuntu WSL 2

This configuration runs Python inside Ubuntu WSL while Ollama runs on Windows.

### 8.1. Prerequisites

- Windows with WSL 2.
- Ubuntu installed through WSL.
- Python 3.14 installed in Ubuntu.
- Git.
- Visual Studio Code with the WSL extension.
- Ollama installed on Windows.

### 8.2. Clone the Repository

In the Ubuntu terminal:

```bash
git clone <REPOSITORY_URL>
cd time-series-agent
```

### 8.3. Create a Virtual Environment

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
```

### 8.4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 8.5. Configure Ollama on Windows

Ollama must listen on an interface accessible from WSL.

First, stop any existing Ollama instance.

In Windows PowerShell:

```powershell
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

Keep the terminal open while using the application.

Verify the listening address:

```powershell
Get-NetTCPConnection -LocalPort 11434 -State Listen
```

**Security warning:** Binding Ollama to `0.0.0.0` can expose its unauthenticated API to other devices. Restrict incoming connections using Windows Firewall and use this configuration only on trusted networks.

### 8.6. Configure Access from WSL

In Ubuntu:

```bash
ip route show default
```

Identify the Windows gateway address.

Configure the environment:

```bash
export OLLAMA_HOST=http://<WSL_GATEWAY_IP>:11434
export OLLAMA_MODEL=qwen3:8b
```

Example:

```bash
export OLLAMA_HOST=http://172.28.64.1:11434
```

The gateway IP address may change after restarting WSL.

### 8.7. Test the Connection

```bash
curl "$OLLAMA_HOST/api/tags"
```

A successful response returns JSON.

Run the Python test script:

```bash
python main.py
```

## 9. Docker Setup

Docker provides a reproducible application environment for the development team.

The selected architecture runs Ollama directly on Windows and the Python application inside a Docker container.

### 9.1. Install Docker Desktop

Download Docker Desktop:

https://www.docker.com/products/docker-desktop/

Enable:

- WSL 2 based engine.
- WSL integration for Ubuntu.

Verify from WSL:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

### 9.2. Docker Architecture

```text
Windows Host
│
├── Ollama
│   └── Qwen3:8b
│
└── Docker Desktop
    └── Python Container
        ├── LangGraph Agent
        ├── LangChain Tools
        ├── Analysis Services
        └── Streamlit Interface
```

### 9.3. Connect Docker to Ollama

Docker Desktop provides the hostname:

```text
host.docker.internal
```

Test connectivity:

```bash
docker run --rm curlimages/curl \
  http://host.docker.internal:11434/api/tags
```

A successful JSON response confirms that Docker can communicate with Ollama.

### 9.4. Dockerfile

The `Dockerfile` defines the application image.

Example:

```dockerfile
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

COPY main.py ./

CMD ["python", "main.py"]
```

All dependencies must be compatible with Python 3.14.

### 9.5. Docker Compose

The `compose.yaml` file configures the application service.

```yaml
services:
  agent:
    build:
      context: .
      dockerfile: Dockerfile

    environment:
      OLLAMA_HOST: http://host.docker.internal:11434
      OLLAMA_MODEL: qwen3:8b

    stdin_open: true
    tty: true
```

### 9.6. Build and Run

Build the image:

```bash
docker compose build
```

Run the agent:

```bash
docker compose run --rm agent
```

When the Streamlit interface is implemented, the Docker configuration can be updated to expose port `8501`.

### 9.7. Docker Hub Connectivity Issues

Downloading Docker images requires access to Docker Hub.

If Docker reports:

```text
Connection reset by peer
```

while connecting to `auth.docker.io`, check:

- Internet connectivity.
- VPN or proxy configuration.
- Windows Firewall.
- Docker Desktop proxy settings.
- Network restrictions.

This error is generally unrelated to Python or Qwen3:8b.

## 10. Example User Requests

Once implemented, the agent should support requests such as:

- "Load this CSV file and summarize its contents."
- "Plot the temperature measurements over time."
- "Calculate the mean and standard deviation."
- "Identify anomalies in the sensor readings."
- "Does this time series show a seasonal pattern?"
- "Forecast the next seven days."
- "Explain the forecasting results."

The agent will interpret these requests and execute the appropriate analytical tools.

## 11. Tools and Services

The project separates LangChain tools from Python services.

### Services

Services contain the core business logic and operate independently of the LLM.

Examples include:

- Loading and validating CSV datasets.
- Handling missing values.
- Calculating descriptive statistics.
- Detecting anomalies.
- Training forecasting models.
- Generating Plotly figures.

### Tools

LangChain tools expose selected service functions to the conversational agent.

This separation improves modularity, testability, and maintainability.

## 12. Testing

Automated tests are located in the `tests/` directory.

Run tests using:

```bash
python -m pytest
```

Tests should cover:

- Data loading and validation.
- Preprocessing.
- Statistical calculations.
- Anomaly detection.
- Forecasting.
- LangChain tool behavior.
- Agent workflow behavior.
- Error handling.

Unit tests should not require an active Ollama server whenever external LLM calls can be mocked.

## 13. Continuous Integration

GitHub Actions will automate quality checks.

The workflow is defined in:

```text
.github/workflows/tests.yml
```

The CI pipeline is intended to:

1. Set up the Python environment.
2. Install dependencies.
3. Run automated tests.
4. Report failures before changes are merged.

GitHub Actions handles repository automation, while LangGraph handles conversational agent orchestration.

## 14. Development Roadmap

The project is planned over eight weeks.

| Week | Main Objectives |
|---|---|
| 1 | Project setup, architecture, GitHub, Ollama |
| 2 | CSV loading and preprocessing |
| 3 | Statistical analysis and visualization |
| 4 | Anomaly detection |
| 5 | Forecasting and model evaluation |
| 6 | LangChain and LangGraph integration |
| 7 | Streamlit interface and system integration |
| 8 | Testing, Docker, documentation, and demonstration |

The roadmap may evolve according to project progress.

## 15. Collaboration Workflow

Development is organized around Git feature branches and pull requests.

Create a feature branch:

```bash
git checkout -b feature/data-loading
```

Commit changes:

```bash
git add .
git commit -m "feat: implement CSV data loading"
```

Push the branch:

```bash
git push -u origin feature/data-loading
```

Team members should review changes before merging them into the main branch.

### Files Excluded from Version Control

The `.gitignore` file should exclude:

- Python virtual environments.
- Local `.env` files.
- Python caches.
- Temporary files.
- Private datasets.
- Large trained model artifacts.
- Generated outputs.

The `.env.example` file should remain version-controlled to document the required environment variables.

## 16. Project Status

### Validated

- Python development environment in Ubuntu WSL.
- Local Ollama installation on Windows.
- Qwen3:8b model availability.
- Communication between WSL Python and Ollama.
- Access to Ollama from a Docker container.
- Docker Desktop integration with WSL.

### In Progress / Planned

- Complete Docker application image build.
- CSV data loading and preprocessing.
- Statistical analysis and anomaly detection.
- Forecasting models.
- LangChain tool integration.
- LangGraph agent orchestration.
- Streamlit user interface.
- Automated tests and continuous integration.

## 17. License

To be determined by the project team.

---

**Academic project — Conversational AI Agent for Time Series Analysis.**