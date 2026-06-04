<div align="center">
  <img alt="mloom_banner" src="https://github.com/user-attachments/assets/95107e38-6163-408d-8de1-597f7055e9c6" width="80%" />

  <h1>mloom</h1>
  <h3>Machine Learning Loom — <i>Weave your machine learning pipeline.</i></h3>
  <p><b>A lightweight tool for ML/LLM Ops and Orchestration.</b></p>
</div>

---

The project's main goal is to develop a lightweight alternative to industry-standard LLMOps tools for independent developers and researchers.

> **This repository is in early development.** > Many features are either missing or nonexistent. It is not in a functional state.

## Project Overview

This document provides a general overview of the project design, architecture, and goals. The full feature set and comprehensive documentation will be published once development is complete.

## Design Decisions and Architecture

### 1. Packaging and Distribution: "Fat Package" Approach
The platform is packaged as a single codebase containing the SDK, CLI, Streamlit UI, and FastAPI server. Installation footprints are managed through optional dependencies to suit different environments:
* **Standard Installation (`pip install mloom`):** Installs the client SDK, Streamlit interface, and SQLite drivers. Designed for local development without the overhead of external databases or servers.
* **Server Installation (`pip install "mloom[server]"`):** Installs the FastAPI server, Uvicorn, Pydantic, and PostgreSQL drivers. Used for centralized deployments in team environments.

### 2. Context Switching and Smart Routing
Operational modes are managed dynamically via a global configuration file (`~/.mloom/config.json`) and the CLI. Code execution does not rely on hardcoded database credentials or endpoints:
* **`mloom ui`:** Default mode. Launches the Streamlit interface and reads data directly from the local database (`~/.mloom/local.db`).
* **`mloom server start`:** Launches the FastAPI backend on a server host.
* **`mloom server connect <IP_ADDRESS>`:** Configures the client SDK to stream execution telemetry as HTTP requests to the remote FastAPI server instead of writing to SQLite.
* **`mloom server connect local`:** Switches the client back to isolated local SQLite mode.

### 3. Dashboard-Driven Experimentation
Instead of passive monitoring, the platform supports active experiment management. By wrapping hyperparameters, model configurations, or metrics with placeholder variables, developers can run experiments directly from the dashboard:
* Placeholders defined in code (e.g., dropdown selectors, sliders) are parsed by the Streamlit dashboard.
* The dashboard renders input components dynamically.
* Clicking the run button triggers the target function with the selected inputs via dependency injection.

### 4. Data Access Strategy
Data retrieval adaptively changes based on the active configuration:
* **Local Mode:** Streamlit queries the SQLite database directly via SQL queries, eliminating network latency.
* **Remote Mode:** Streamlit queries read-only HTTP endpoints of the remote FastAPI backend rather than exposing the PostgreSQL database directly to the network.

---

## Directory Structure

```plaintext
mloom-repo/
├── mloom/                  # Main package directory
│   ├── core/               # Configuration and state management
│   ├── db/                 # Database schemas and SQLAlchemy models
│   ├── sdk/                # Decorators and placeholder classes
│   ├── server/             # FastAPI routes and schemas
│   ├── ui/                 # Streamlit UI dashboard code
│   └── cli.py              # CLI context and mode switcher
├── pyproject.toml          # Packaging and dependencies
└── README.md               # Documentation
```

---

## Usage Example

The following example demonstrates how placeholder variables are integrated into functions to enable dynamic configuration and execution through the dashboard:

```python
from mloom import track_llm, variables

# Define placeholders for dynamic execution from the UI
selected_model = variables.ModelSelect(
    name="LLM_Model", 
    options=["gpt-4o", "claude-3", "llama-3"], 
    default="gpt-4o"
)
temperature = variables.FloatSlider(
    name="Temperature", 
    min=0.0, 
    max=1.0, 
    default=0.7
)

@track_run
def run_evaluation(model=selected_model, temp=temperature):
    # The SDK injects values selected in the dashboard at runtime
    pass
```
