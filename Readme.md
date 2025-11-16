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
10. [Streamlit Dashboard](#10-streamlit-dashboard)
11. [Outputs and Persistence](#11-outputs-and-persistence)
12. [Testing and Quality](#12-testing-and-quality)
13. [Project Structure](#13-project-structure)
14. [Deployment and Kaggle Usage](#14-deployment-and-kaggle-usage)
15. [Security and Privacy](#15-security-and-privacy)
16. [Roadmap](#16-roadmap)
17. [Support and Contributions](#17-support-and-contributions)
18. [Dashboard Walkthrough](#18-dashboard-walkthrough)
19. [Customization & Extensibility](#19-customization--extensibility)
20. [Troubleshooting & FAQ](#20-troubleshooting--faq)

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
The default pipeline prefers the multi-facility sample `data/multi_hospital_energy.csv` (falls back to `data/hospital_energy.csv`) and reconciles common column aliases before analysis.

| Canonical Field        | Description                                   | Accepted Aliases              |
|------------------------|-----------------------------------------------|-------------------------------|
| `date`                 | ISO-8601 date or parseable timestamp          | —                             |
| `electricity_kwh`      | Electricity consumption in kilowatt-hours     | `energy_usage_kwh`            |
| `water_gallons`        | Water usage in gallons                        | `water_usage_liters`*         |
| `carbon_emissions_kg`  | Carbon emissions in kilograms of CO₂          | `emissions_kgco2`             |
| `waste_kg`             | Solid waste generated (kilograms)             | —                             |
| `recycling_kg`         | Recyclables collected (kilograms)             | —                             |
| `facility`             | Facility identifier for comparisons           | `facility_id`, `facility_name`|

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
- Place your CSV in `data/` and update the path passed into the orchestrator (defaults to `data/multi_hospital_energy.csv` when available).
- Ensure column names align with the aliases in `agents/data_collector.py` or extend the mapping.

## 10. Streamlit Dashboard
- Launch the interactive UI with:
    ```cmd
    streamlit run streamlit_app.py
    ```
- **Sidebar controls**: toggle LLM usage, choose an LLM provider, and upload a custom CSV. Leave the uploader empty to run against the bundled multi-hospital sample.
- **Overview tab**: metric cards, energy/carbon/water trend charts, facility breakdowns, operational insights, and anomaly spotlights.
- **Analysis tab**: drill into summaries, carbon intensity trends, anomaly details, efficiency metrics, correlations, and facility comparisons.
- **Interventions tab**: impact/timeframe charts, KPI counters, and card-based recommendation gallery with a roadmap load visual.
- **Report tab**: executive summary KPIs, narrative highlights, keyword theme chart, and tiled visualization gallery.
- **Data & History tab**: filtered dataset preview, distribution charts, numeric snapshot, energy/carbon correlations, session replay timeline, and run metadata.
- All charts respect the new Streamlit width API (`width="stretch"`) to remain forward-compatible after 2025.
- Facility and timeline filters appear above the tabs so you can curate stories for specific hospitals or date windows without rerunning the workflow.
- Every run still writes plots to `outputs/`, logs to `logs/`, and memories to `data/long_term_memory.json` so results remain reproducible outside the app.

## 11. Outputs and Persistence
- **Visualizations**: PNG charts saved to `outputs/` (directory is git-ignored).
- **Logs**: Detailed execution logs in `logs/workflow.log` for diagnostics.
- **Long-term memory**: Run summaries appended to `data/long_term_memory.json` with timestamps and key metrics.
- **Console output**: High-level progress and highlights printed to stdout.

## 12. Testing and Quality
- Automated tests live in `tests/` and cover agent logic plus the end-to-end workflow.
- Execute the suite with:
    ```cmd
    .venv\Scripts\pytest -q
    ```
- Suggested CI routine: install dependencies, run `pytest`, then execute `python main.py` once with `ENABLE_LLM=0` to validate the deterministic path.

## 13. Project Structure
```
├─ agents/
│  ├─ analysis_agent.py
│  ├─ data_collector.py
│  ├─ intervention_planner.py
│  └─ report_generator.py
├─ data/
│  ├─ hospital_energy.csv
│  ├─ multi_hospital_energy.csv
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

## 14. Deployment and Kaggle Usage
- Set `ENABLE_LLM=0`; Kaggle kernels generally block external API calls.
- Upload datasets via the Kaggle Data tab so they appear under `/kaggle/input/<dataset>/`.
- Write outputs to `/kaggle/working/outputs` if you need downloadable artifacts.
- Pin package versions in `requirements.txt` to keep kernels reproducible.

## 15. Security and Privacy
- `.gitignore` already excludes `.env`, `logs/`, `outputs/`, and other local artifacts.
- Never commit API keys. Rotate any credential immediately if exposed.
- For CI/CD, store secrets in the platform’s vault and inject them at runtime only when LLM features are required.

## 16. Roadmap
- Add forecasting modules for energy and emissions projections.
- Expand anomaly detection with ML-based techniques.
- Ship a Kaggle-ready notebook template and GitHub Actions workflow for automated testing.
- Introduce optional dashboard front-end for real-time monitoring.

## 17. Support and Contributions
- Open issues or pull requests at `https://github.com/Santosh-Reddy1310/Hospi-Sus-Agent`.
- Follow PEP 8, include type hints where practical, and add tests for new behavior.
- A `LICENSE` file (MIT recommended) can be added based on deployment needs; request it if required.

## 18. Dashboard Walkthrough
- **Prerequisites**: activate the virtual environment, ensure `streamlit` is installed, and run `streamlit run streamlit_app.py`.
- **Initial state**: the sidebar defaults to the bundled multi-hospital dataset with LLM agents disabled for deterministic execution.
- **Filters ribbon**: facility multiselect and timeline slider appear above the main tabs; both derive their options from the currently loaded dataset and update every visualization in real time.
- **Overview tab**:
    - Metric deck summarises record counts, energy, carbon, water, anomaly counts, and efficiency indicators.
    - Trend block plots month-end energy, carbon, and water trajectories plus a facility energy leaderboard.
    - Operational pulse charts highlight HVAC usage distribution and the relationship between operational hours and energy load.
    - Alerts block visualises anomaly counts and thresholds to direct attention to problematic metrics.
- **Analysis tab**:
    - Summary section provides bar charts for top-line totals and auto-generated facility comparisons.
    - Trends page exposes direction-of-change metrics and carbon intensity charts, including a monthly intensity timeline.
    - Anomalies page restates the alert visuals with tabular context and thresholds for auditors.
    - Efficiency page blends metric cards with a correlation heatmap to surface co-movement across numeric features.
- **Interventions tab**:
    - KPI strip counts recommendations, focus areas, and high-impact items.
    - Impact and timeline bar charts reveal recommended workload distribution.
    - Recommendation gallery lists each action with impact, cost, timeline, savings, and category metadata for ease of review.
    - Roadmap histogram summarises effort per phase, followed by phase-specific bullet lists.
- **Report tab**:
    - Executive summary automatically converts structured statistics into charts when available, falling back to a textual summary when not.
    - Narrative is reformatted into digestible highlights and paired with a keyword frequency chart so readers can spot dominant themes quickly.
    - Visual assets display side by side using the new `width="stretch"` parameter for Streamlit 2026 compatibility.
- **Data & History tab**:
    - Headline metrics show rows, facilities, and temporal coverage after filtering.
    - Distribution charts cover facility share, HVAC mix, numeric summaries, weekly energy trends, and energy-versus-carbon scatter relationships.
    - Session history leverages `st.status` to replay the workflow steps; a plain list fallback is used if the Streamlit runtime predates the widget.
    - Metadata block exposes workflow identifiers, filtered record counts, and execution timestamps for traceability.
- **Download button**: exports the filtered dataset view so analysts can audit specific slices offline without rerunning the workflow.

## 19. Customization & Extensibility
- **Adding new data fields**:
    - Extend alias mappings in `agents/data_collector.py` to recognise additional columns and perform unit conversions.
    - Update `_derive_analysis_from_dataframe` in `streamlit_app.py` to compute summary statistics for new metrics so they appear across cards and charts.
- **Expanding the analysis layer**:
    - Introduce new computations in `AnalysisAgent` then propagate results through the orchestrator; the Streamlit app consumes whatever the orchestrator returns and can surface new entries automatically.
    - Use the correlation heatmap as a template for additional diagnostics such as regression coefficients or clustering outputs.
- **Custom interventions**:
    - Modify `InterventionPlannerAgent` to map analysis signals to bespoke recommendations, categories, and estimated savings.
    - The dashboard automatically renders any additional attributes you include in the recommendation dictionaries.
- **LLM augmentation**:
    - Enable `ENABLE_LLM=1` with the provider API key to swap local fallbacks for CrewAI agents that can craft richer narratives and interventions.
    - Consider adding provider-specific prompt templates or guardrails inside the agent classes if you need domain-specific tone or terminology.
- **Deploying beyond local machines**:
    - Package the Streamlit app as a container, expose environment variables for configuration, mount a persistent volume for `logs/` and `outputs/`, and point `SAMPLE_DATA_PATH` to a cloud-hosted dataset if required.
    - Integrate with CI pipelines by running `pytest`, the deterministic workflow, and a headless Streamlit smoke test (e.g., `streamlit hello --server.headless true`).

## 20. Troubleshooting & FAQ
- **Why does the dashboard show “Sample dataset unavailable”?** Ensure `data/multi_hospital_energy.csv` or `data/hospital_energy.csv` exists. Upload a CSV or supply a valid path in the sidebar if both are missing.
- **Streamlit warns about deprecated parameters**: all width-related options now use `width="stretch"` or `width="content"`. If you extend components, follow the same pattern to stay compatible with post-2025 Streamlit releases.
- **Missing visualizations**: confirm the `outputs/` folder is writable. The orchestrator catches plotting errors, but permission issues can still prevent files from being saved.
- **LLM agent initialization failed**: double-check `ENABLE_LLM`, the provider API key, and outbound network access. The orchestrator logs a warning and reverts to local agents automatically when initialization fails.
- **How do I reset session history in Streamlit?** Click “Rerun” in the Streamlit UI or clear `st.session_state["workflow_results"]` via the developer console. Cached results persist until you explicitly run a new analysis.
- **Where are logs stored?** All runs append to `logs/workflow.log`. Tail the file while Streamlit is running to monitor the orchestrator in real time.
- **Dataset has different facility names each month**: enable the facility filter to focus on a single hospital, or harmonise facility naming upstream before ingestion to avoid splitting metrics across similar labels.
- **Can I schedule automated runs?** Yes. Invoke `python main.py` from Task Scheduler, cron, or a CI runner. Generated visuals and JSON outputs will stack inside `outputs/` and `data/long_term_memory.json` for later review.
