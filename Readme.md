# Healthcare Sustainability AI Agent System

> End-to-end multi-agent workflow that helps hospitals measure, analyze, and improve environmental sustainability metrics.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Core Capabilities](#2-core-capabilities)
3. [Architecture](#3-architecture)
4. [Data Model](#4-data-model)
5. [Workflow Lifecycle](#5-workflow-lifecycle)
6. [Agent Responsibilities](#6-agent-responsibilities)
7. [Getting Started](#7-getting-started)
8. [Configuration](#8-configuration)
9. [Running the System](#9-running-the-system)
10. [Outputs and Persistence](#10-outputs-and-persistence)
11. [Testing and Quality](#11-testing-and-quality)
12. [Project Structure](#12-project-structure)
13. [Deployment and Kaggle Usage](#13-deployment-and-kaggle-usage)
14. [Security and Privacy](#14-security-and-privacy)
15. [Roadmap](#15-roadmap)
16. [Support and Contributions](#16-support-and-contributions)

## 1. Project Overview
- **Mission**: empower healthcare sustainability leaders with automated insights so they can curb energy usage, emissions, water consumption, and waste.
- **Scope**: orchestrates data ingestion, analytics, visualization, and intervention planning for one or more facilities.
- **Design Goals**: reproducible runs without LLM access, optional LLM augmentation, strong logging and persistence, easy packaging for Kaggle or CI environments.

## 2. Core Capabilities
- Automated extraction and normalization of operational metrics from CSV sources.
- Statistical analysis, trend detection, anomaly surfacing, and efficiency benchmarking.
- Narrative and visual sustainability reports saved to disk for stakeholders.
- Intervention roadmap generator that prioritizes high-impact sustainability actions.
- JSON-backed memory layer that records historical run summaries for auditing.

## 3. Architecture
```
┌───────────────────────────────┐
│      Interface Options        │
│  CLI • Notebook • Kaggle UI   │
└──────────────┬────────────────┘
               │
┌──────────────┴──────────────┐
│      Orchestration Core     │
│    (main_workflow.py)       │
└──────────────┬──────────────┘
               │
┌──────────────┴──────────────┐
│         Agent Layer         │
│ Data ▪ Analysis ▪ Reports ▪ │
│       Interventions         │
└──────────────┬──────────────┘
               │
└──────────────┴──────────────┐
│     Memory & Persistence    │
│ Session ▪ Long-term JSON    │
└─────────────────────────────┘
```

## 4. Data Model
The default pipeline ingests `data/hospital_energy.csv` and reconciles common column aliases.

| Canonical Field        | Description                                   | Accepted Aliases              |
|------------------------|-----------------------------------------------|-------------------------------|
| `date`                 | ISO-8601 date or parseable timestamp          | —                             |
| `electricity_kwh`      | Electricity consumption in kilowatt-hours     | `energy_usage_kwh`            |
| `water_gallons`        | Water usage in gallons                        | `water_usage_liters`*         |
| `carbon_emissions_kg`  | Carbon emissions in kilograms of CO₂          | `emissions_kgco2`             |
| `waste_kg`             | Solid waste generated (kilograms)             | —                             |
| `recycling_kg`         | Recyclables collected (kilograms)             | —                             |
| `facility_id`          | Optional facility identifier for comparisons  | —                             |

*`water_usage_liters` is automatically converted to gallons and the original column removed to prevent duplication.

## 5. Workflow Lifecycle
1. **Data ingestion** – `DataCollectorAgent` loads, validates, and enriches the dataset.
2. **Analytics** – `AnalysisAgent` computes statistics, trends, anomalies, and intensity metrics.
3. **Reporting** – `ReportGeneratorAgent` produces executive summaries and PNG visualizations.
4. **Intervention design** – `InterventionPlannerAgent` recommends prioritized sustainability actions.
5. **Persistence** – results, plots, and run metadata are written to `outputs/`, `logs/`, and `data/long_term_memory.json`.

## 6. Agent Responsibilities
Each agent follows a `run(payload: dict) -> dict` contract and returns structured results with status metadata.

### 6.1 DataCollectorAgent (`agents/data_collector.py`)
- Normalizes schema, handles alias mapping, performs safe unit conversions, and adds derived columns such as `day_of_week`.
- Emits a validation summary that flags missing values, duplicates, and schema issues.

### 6.2 AnalysisAgent (`agents/analysis_agent.py`)
- Generates summary statistics, trend comparisons (first vs last period), anomaly detection, and efficiency benchmarks.
- Supports multi-facility comparisons when `facility_id` is present.

### 6.3 ReportGeneratorAgent (`agents/report_generator.py`)
- Crafts an executive summary, detailed narrative, and matplotlib-based charts saved in `outputs/`.
- Includes graceful fallbacks for environments lacking extra plotting styles.

### 6.4 InterventionPlannerAgent (`agents/intervention_planner.py`)
- Converts analysis findings into prioritized action plans with estimated impact, effort, and phase recommendations.
- Produces an implementation roadmap separating quick wins from longer-term initiatives.

## 7. Getting Started
### 7.1 Prerequisites
- Python 3.10 or newer (3.11 and 3.12 validated)
- Git (optional but recommended)
- Windows instructions below use `cmd.exe`; adjust for PowerShell or Unix shells as needed

### 7.2 Installation
```cmd
git clone https://github.com/Santosh-Reddy1310/Hospi-Sus-Agent.git
cd Hospi-Sus-Agent
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## 8. Configuration
Environment variables control LLM usage, providers, and runtime behavior. Use the included `scripts/save_key.py` helper to create a local `.env` (ignored by git).

| Variable           | Accepted Values         | Purpose                                      |
|--------------------|-------------------------|----------------------------------------------|
| `ENABLE_LLM`       | `0` (default) or `1`     | Toggle LLM-powered CrewAI agents             |
| `LLM_PROVIDER`     | `openai`, `groq`         | Selects provider when LLMs are enabled       |
| `OPENAI_API_KEY`   | Provider key             | Required when `ENABLE_LLM=1` and provider set|
| `GROQ_API_KEY`     | Provider key             | Same as above                                |

Unset keys or `ENABLE_LLM=0` automatically trigger deterministic local agents so the workflow remains offline-friendly.

## 9. Running the System
### 9.1 Deterministic run (recommended for first execution, CI, and Kaggle)
```cmd
set ENABLE_LLM=0
.venv\Scripts\python main.py
```

### 9.2 LLM-augmented run
```cmd
set ENABLE_LLM=1
set LLM_PROVIDER=groq
set GROQ_API_KEY=your_key_here
.venv\Scripts\python main.py
```

### 9.3 Using a custom dataset
- Place your CSV in `data/` and update the path passed into the orchestrator (defaults to `data/hospital_energy.csv`).
- Ensure column names align with the aliases in `agents/data_collector.py` or extend the mapping.

## 10. Outputs and Persistence
- **Visualizations**: PNG charts saved to `outputs/` (directory is git-ignored).
- **Logs**: Detailed execution logs in `logs/workflow.log` for diagnostics.
- **Long-term memory**: Run summaries appended to `data/long_term_memory.json` with timestamps and key metrics.
- **Console output**: High-level progress and highlights printed to stdout.

## 11. Testing and Quality
- Automated tests live in `tests/` and cover agent logic plus the end-to-end workflow.
- Execute the suite with:
    ```cmd
    .venv\Scripts\pytest -q
    ```
- Suggested CI routine: install dependencies, run `pytest`, then execute `python main.py` once with `ENABLE_LLM=0` to validate the deterministic path.

## 12. Project Structure
```
├─ agents/
│  ├─ analysis_agent.py
│  ├─ data_collector.py
│  ├─ intervention_planner.py
│  └─ report_generator.py
├─ data/
│  ├─ hospital_energy.csv
│  └─ long_term_memory.json
├─ logs/                # created at runtime
├─ outputs/             # generated visuals (git-ignored)
├─ scripts/
│  └─ save_key.py
├─ tests/
│  ├─ test_agents.py
│  └─ test_workflow.py
├─ main.py              # CLI entry point
└─ main_workflow.py     # Orchestrator and agent wiring
```

## 13. Deployment and Kaggle Usage
- Set `ENABLE_LLM=0`; Kaggle kernels generally block external API calls.
- Upload datasets via the Kaggle Data tab so they appear under `/kaggle/input/<dataset>/`.
- Write outputs to `/kaggle/working/outputs` if you need downloadable artifacts.
- Pin package versions in `requirements.txt` to keep kernels reproducible.

## 14. Security and Privacy
- `.gitignore` already excludes `.env`, `logs/`, `outputs/`, and other local artifacts.
- Never commit API keys. Rotate any credential immediately if exposed.
- For CI/CD, store secrets in the platform’s vault and inject them at runtime only when LLM features are required.

## 15. Roadmap
- Add forecasting modules for energy and emissions projections.
- Expand anomaly detection with ML-based techniques.
- Ship a Kaggle-ready notebook template and GitHub Actions workflow for automated testing.
- Introduce optional dashboard front-end for real-time monitoring.

## 16. Support and Contributions
- Open issues or pull requests at `https://github.com/Santosh-Reddy1310/Hospi-Sus-Agent`.
- Follow PEP 8, include type hints where practical, and add tests for new behavior.
- A `LICENSE` file (MIT recommended) can be added based on deployment needs; request it if required.
