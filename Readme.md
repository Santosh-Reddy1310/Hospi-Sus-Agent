# Healthcare Sustainability AI Agent System

## Complete Project Documentation

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technical Specifications](#technical-specifications)
4. [Implementation Guide](#implementation-guide)
5. [Agent Specifications](#agent-specifications)
6. [Testing & Evaluation](#testing--evaluation)
7. [Deployment Instructions](#deployment-instructions)
8. [Future Enhancements](#future-enhancements)

---

## 1. Executive Summary

### 1.1 Project Overview
The Healthcare Sustainability AI Agent System is a multi-agent application designed to help healthcare facilities monitor, analyze, and optimize their environmental sustainability practices. The system autonomously collects data, performs analysis, generates comprehensive reports, and provides actionable intervention recommendations.

### 1.2 Problem Statement
Healthcare facilities are significant contributors to carbon emissions and waste generation. Manual sustainability tracking is time-consuming, error-prone, and often lacks actionable insights. This system automates the entire sustainability management workflow.

### 1.3 Solution Approach
A collaborative multi-agent system where specialized AI agents work together:
- **Data Collector Agent**: Gathers sustainability metrics
- **Analysis Agent**: Processes and identifies patterns
- **Report Generator Agent**: Creates visual and textual reports
- **Intervention Planner Agent**: Recommends optimization strategies

### 1.4 Key Features
- Automated data collection from multiple sources
- Real-time sustainability analysis
- Visual dashboard generation
- Actionable recommendations based on industry best practices
- Session and long-term memory for context retention
- Comprehensive logging and evaluation metrics

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                  (Kaggle Jupyter Notebook)                   │
└──────────────────────────┬──────────────────────────────────┘
                        # Hospi-Sus-Agent — Project Documentation

                        Badges: [CI] [PyPI] [License]  
                        (Replace these with actual badges when CI / packaging are added.)

                        This repository contains a modular, multi-agent system for hospital sustainability analysis. It is designed for both interactive experimentation (notebooks / Kaggle) and automated batch runs. The system ingests operational data (energy, water, waste), performs analyses (statistics, trends, anomalies), produces narrative and visual reports, and recommends prioritized interventions.

                        Table of contents

                        - 1. Project overview
                        - 2. Quick start (local)
                        - 3. Configuration and environment variables
                        - 4. Data format and schema
                        - 5. Agents: API and behavior
                        - 6. Orchestrator and workflow
                        - 7. Outputs, persistence, and memory
                        - 8. Development & testing
                        - 9. Packaging for Kaggle
                        - 10. Security, secrets, and safe development
                        - 11. Contributing, license, and contact

                        ---

                        1. Project overview
                        -------------------
                        Purpose: provide operational analytics and recommendations to help hospitals reduce energy, water, waste, and carbon emissions. The system targets facility managers, sustainability teams, and researchers.

                        Key goals:

                        - Enable reproducible, offline runs (no external API required) for Kaggle and CI.
                        - Allow optional LLM-powered agents for richer narrative outputs when credentials are provided.
                        - Provide a clear extension path: add data sources, custom analyses, or alternative output formats.

                        2. Quick start (local)
                        ----------------------
                        Requirements: Python 3.10+ (3.11/3.12 tested), Git.

                        Install and run with local fallbacks (recommended for first run):

                        ```cmd
                        python -m venv .venv
                        .venv\Scripts\python -m pip install --upgrade pip
                        .venv\Scripts\python -m pip install -r requirements.txt

                        set ENABLE_LLM=0
                        .venv\Scripts\python main.py
                        ```

                        Notes:

                        - `ENABLE_LLM=0` forces local deterministic agents (no network calls). Use this in CI and Kaggle.
                        - To enable LLM agents set `ENABLE_LLM=1` and the provider env var (see Section 3).

                        3. Configuration and environment variables
                        ----------------------------------------
                        Runtime options are configured through environment variables. The project uses a small set of well-named variables.

                        - `ENABLE_LLM` (0 or 1) — Enable LLM-powered agents. Default: `0`.
                        - `LLM_PROVIDER` — `openai` or `groq`. Default: `openai`.
                        - `OPENAI_API_KEY` / `GROQ_API_KEY` — Provider-specific keys. Provide only in your local environment or CI secrets manager.

                        Examples (Windows cmd.exe):

                        ```cmd
                        set ENABLE_LLM=1
                        set LLM_PROVIDER=openai
                        set OPENAI_API_KEY=sk-xxx
                        .venv\Scripts\python main.py
                        ```

                        The project includes `scripts/save_key.py` which prompts for a key and writes a local `.env` file (this file is ignored by git by default). Do not commit `.env`.

                        4. Data format and schema
                        -------------------------
                        The DataCollector accepts CSV files with common hospital operational fields. The DataCollector implements an alias mapping to support variations in column names.

                        Canonical column names used internally (preferred):

                        - `date` — ISO date string or parseable date
                        - `electricity_kwh` — energy in kWh
                        - `water_gallons` — water usage in gallons (if `water_liters` present it is converted)
                        - `carbon_emissions_kg` — emissions in kg CO2
                        - `waste_kg`, `recycling_kg` — waste metrics
                        - `facility_id` — optional string identifier

                        Common aliases supported (examples):

                        - `energy_usage_kwh` -> `electricity_kwh`
                        - `water_usage_liters` -> `water_gallons` (converted)
                        - `emissions_kgco2` -> `carbon_emissions_kg`

                        If your dataset uses different names, either rename columns before running or extend `agents/data_collector.py` alias mapping.

                        5. Agents: API and behavior
                        ---------------------------
                        Each agent exposes a single main entry `run(inputs: dict) -> dict` where `inputs` is a context dictionary and the return value is a structured result including `status` and output data.

                        Agent contract (short):

                        - Input: `dict` with keys documented per agent (or the orchestrator will populate defaults).
                        - Output: `dict` with at least `status: 'success'|'error'` and a `data` key with structured results.

                        Detailed agent descriptions:

                        - DataCollectorAgent (`agents/data_collector.py`)
                            - Purpose: load CSV or API, validate schema, alias & normalize columns, add derived fields like `day_of_week`, and return `pandas.DataFrame`-serializable data plus a validation report.
                            - Important behavior: will not overwrite canonical columns; maps aliases only when canonical is absent and drops original columns after safe conversion to avoid duplicates.

                        - AnalysisAgent (`agents/analysis_agent.py`)
                            - Purpose: compute summary statistics, detect trends and anomalies, compute efficiency metrics (e.g., energy per bed or per floor area), and compute carbon intensity metrics.
                            - Output: nested dict with `summary_statistics`, `trend_analysis`, `anomalies` and `efficiency_metrics`.

                        - ReportGeneratorAgent (`agents/report_generator.py`)
                            - Purpose: generate plain-text executive summary, human-readable report, and visualizations (PNG files saved to `outputs/`).
                            - Output: `report_text`, `executive_summary`, and `visual_paths`.

                        - InterventionPlannerAgent (`agents/intervention_planner.py`)
                            - Purpose: suggest prioritized interventions and an implementation roadmap based on analysis outputs. Ranks interventions by impact and cost heuristics.

                        6. Orchestrator and workflow
                        ----------------------------
                        `main_workflow.py` coordinates the following steps:

                        1. Data ingestion (DataCollector)
                        2. Analysis (AnalysisAgent)
                        3. Report generation (ReportGenerator)
                        4. Intervention planning (InterventionPlanner)
                        5. Persist run snapshot to `data/long_term_memory.json`

                        The orchestrator is robust: when `ENABLE_LLM=0` or provider keys are missing it uses local deterministic agent implementations that do not make network calls. This makes it CI- and Kaggle-friendly.

                        Run the orchestrator directly:

                        ```cmd
                        set ENABLE_LLM=0
                        .venv\Scripts\python main.py
                        ```

                        7. Outputs, persistence and memory
                        ---------------------------------
                        - `outputs/` — PNG visualizations and any generated attachments. This folder is git-ignored by default.
                        - `data/long_term_memory.json` — JSON file storing run snapshots (timestamp, analysis, paths to visuals). Useful for tracking progress across runs.
                        - `logs/workflow.log` — log file with detailed run information.

                        8. Development & testing
                        ------------------------
                        Run unit tests with `pytest` (recommended inside `.venv`).

                        ```cmd
                        .venv\Scripts\python -m pip install -r requirements-dev.txt  # if present
                        .venv\Scripts\pytest -q
                        ```

                        Suggested CI (GitHub Actions):

                        - On push/PR: set `ENABLE_LLM=0`, install dependencies, run `pytest`, and run `python main.py` once to validate end-to-end non-LLM behavior.

                        9. Packaging for Kaggle
                        ----------------------
                        Tips to prepare a Kaggle Notebook / kernel:

                        1. Ensure `ENABLE_LLM=0` in the notebook environment.
                        2. Attach the dataset to the kernel; CSV will be available at `/kaggle/input/<dataset>/hospital_energy.csv`.
                        3. Write `outputs/` to `/kaggle/working/outputs` to allow downloads.
                        4. Keep external network calls disabled.

                        Example notebook cell to run the orchestrator inside Kaggle:

                        ```python
                        import os
                        os.environ['ENABLE_LLM'] = '0'
                        !python main.py
                        ```

                        10. Security, secrets, and safe development
                        ----------------------------------------
                        - Never commit API keys or `.env` to the repository. `.gitignore` excludes `.env` by default.
                        - Use `scripts/save_key.py` to store a key locally for development only (it writes `.env` which is ignored).
                        - For CI, store provider keys in the CI secrets manager and ensure workflows only enable `ENABLE_LLM=1` when secrets are present.

                        11. Contributing, license, and contact
                        -------------------------------------
                        Contributing:

                        - Open issues for bugs or feature requests.
                        - Create small focused PRs; add tests for new behavior.

                        License:

                        - Add a `LICENSE` file (MIT recommended). If you'd like, I can add a draft MIT license now.

                        Contact:

                        - Repository: `https://github.com/Santosh-Reddy1310/Hospi-Sus-Agent`
                        - Open an issue or PR for questions and contributions.

                        Appendix: file map

                        - `main.py` — entrypoint invoking the orchestrator
                        - `main_workflow.py` — orchestrator and local fallback logic
                        - `agents/` — implementation files for each agent
                        - `data/` — sample dataset and `long_term_memory.json`
                        - `outputs/` — generated visualizations (ignored)
                        - `logs/` — runtime logs (ignored)
                        - `scripts/save_key.py` — interactive helper to save env keys locally (ignored by git)

                        Next steps I can take for you
                        ----------------------------
                        - Add a GitHub Actions workflow that runs `pytest` and a non-LLM full run on PRs/pushes.
                        - Produce a Kaggle-ready notebook `capstone_agent_kaggle.ipynb` tuned for `/kaggle/input` and `/kaggle/working/outputs`.
                        - Add a `LICENSE` (MIT) and `CONTRIBUTING.md` template.

                        Tell me which of these you'd like me to do next. I can also commit and push this README update to `origin/main` if you want me to.
                        ---

                        Project overview
                        ----------------
                        The Hospi-Sus-Agent is designed to help facility managers and sustainability teams:

                        - Ingest operational data (energy, water, waste) from CSVs or APIs
                        - Produce automated analyses: summary statistics, trends, anomalies
                        - Generate visual and narrative reports suitable for leadership
                        - Propose prioritized, actionable interventions and an implementation roadmap

                        The system is intentionally modular so components can be replaced, extended, or run in restricted environments (e.g., Kaggle kernels) without network access.

                        Architecture and components
                        ---------------------------
                        High-level layers:

                        - Orchestrator (main_workflow.py): coordinates the entire workflow and persists results
                        - Agent layer: four specialized agents
                            - DataCollectorAgent — ingestion, validation, enrichment
                            - AnalysisAgent — stats, trends, anomalies, comparisons
                            - ReportGeneratorAgent — text & visual reporting
                            - InterventionPlannerAgent — recommendations and roadmap
                        - Memory layer: SessionMemory (ephemeral) and LongTermMemory (JSON persistence)
                        - Utilities: plotting, evaluation, helpers

                        Design choices:

                        - Agents are implemented as CrewAI-compatible classes when LLM is enabled, but the orchestrator provides local, deterministic fallbacks so runs are reproducible without API keys.
                        - Data enrichment normalizes common column name variants (e.g., `energy_usage_kwh` → `electricity_kwh`) and performs safe unit conversions.
                        - Persistent outputs are stored under `outputs/` (visuals) and `data/long_term_memory.json` (JSON snapshots).

                        Installation & quick start (local)
                        ---------------------------------
                        Prerequisites: Python 3.10+ recommended. Create a virtual environment and install dependencies.

                        Windows (cmd.exe):

                        ```cmd
                        python -m venv .venv
                        .venv\Scripts\python -m pip install --upgrade pip
                        .venv\Scripts\python -m pip install -r requirements.txt
                        ```

                        Run the pipeline using local fallbacks (no API keys required):

                        ```cmd
                        set ENABLE_LLM=0
                        .venv\Scripts\python main.py
                        ```

                        If you want LLM-powered agents (optional), set `ENABLE_LLM=1` and provide the provider-specific API key as described in the Security section.

                        Running (examples)
                        -------------------
                        1) Full orchestrator (default):

                        ```cmd
                        set ENABLE_LLM=0           # or 1 to enable LLMs
                        set LLM_PROVIDER=groq      # or openai
                        set GROQ_API_KEY=...       # if using groq and ENABLE_LLM=1
                        .venv\Scripts\python main.py
                        ```

                        2) Use the included local helper to store keys in `.env` (ignored by git):

                        ```cmd
                        .venv\Scripts\python scripts\save_key.py
                        ```

                        Data format and sample dataset
                        ------------------------------
                        The sample CSV `data/hospital_energy.csv` contains daily records with these common fields:

                        - `date` — ISO date
                        - `energy_usage_kwh` or `electricity_kwh` — daily electricity usage in kWh
                        - `water_usage_liters` or `water_gallons` — water usage (liters will be converted to gallons)
                        - `emissions_kgco2` or `carbon_emissions_kg` — calculated emissions in kg CO2
                        - `waste_kg`, `recycling_kg` — waste and recycling weights
                        - `facility_id` — optional facility identifier

                        The DataCollector normalizes known variants (aliases) and adds derived columns when possible (e.g., `total_energy_mwh`, `recycling_rate`). If your dataset uses different column names, either rename them or extend the alias mapping in `agents/data_collector.py`.

                        Agents — design and details
                        ---------------------------
                        Each agent focuses on a single concern. Agents expose a `run(inputs: dict)` method and return a structured dict (`status`, keys, and data). Below are concise descriptions and internal behavior.

                        1) DataCollectorAgent
                        - Responsibilities: load CSV or API, validate schema, normalize columns, enrich data (date parsing, day_of_week), and return both data and a validation report.
                        - Key functions: `load_csv_data(filepath)`, `validate_data(df)`, `enrich_data(df)`.
                        - Behavior notes: the agent will not overwrite canonical columns; instead it maps aliases only when the canonical name is not present. Unit conversions (e.g., liters→gallons) are safe and the original liters column is dropped to avoid duplicate column names.

                        2) AnalysisAgent
                        - Responsibilities: compute summary statistics (mean, median, std, min, max, total), identify time trends (percentage change first→last), detect anomalies using statistical thresholds, compare facilities, and compute efficiency metrics like energy intensity.
                        - Output: a nested `analysis` dict containing `summary_statistics`, `trend_analysis`, `anomaly_detection`, `facility_comparison`, `carbon_intensity`, `efficiency_metrics`.

                        3) ReportGeneratorAgent
                        - Responsibilities: produce an executive summary (plain text), a longer textual report, and visualizations. Visualizations are saved to `outputs/` as PNG files and returned as a list of paths.
                        - Notes: plotting is robust to environments that lack optional styles (tries `seaborn` then falls back to `ggplot` or defaults).

                        4) InterventionPlannerAgent
                        - Responsibilities: produce prioritized recommendations and a phased implementation roadmap. The planner uses analysis outputs to suggest quick wins and further investigations for anomalies.

                        Memory, persistence and outputs
                        -------------------------------
                        - SessionMemory: lightweight, kept in-process for the run, useful for debugging and short-term context.
                        - LongTermMemory: JSON-backed store at `data/long_term_memory.json` which remembers workflow runs with timestamps. Use this to persist results across runs.
                        - Visual outputs: PNGs are placed in `outputs/`. Long-term memory references these file paths.

                        Logging, diagnostics and troubleshooting
                        --------------------------------------
                        - Logging: configured in `main_workflow.py` to write to `logs/workflow.log` and stdout. `logs/` is in `.gitignore` by default.
                        - Common issues:
                            - CrewAI import / LLM errors: ensure `ENABLE_LLM=0` or set provider key env vars.
                            - Duplicate columns warning: occurs when input CSV contains both original and normalized columns (DataCollector now prevents this).
                            - Plot style errors: matplotlib style fallback logic is present.

                        Security and secrets handling (GROQ/OpenAI keys)
                        -----------------------------------------------
                        This project intentionally avoids storing secrets in code or repository files.

                        - `.env.example` shows the environment variables you can set locally (do NOT copy real keys into the repo).
                        - `.gitignore` includes `.env` to prevent accidental commits.
                        - Use the included helper `scripts/save_key.py` to safely write a local `.env` (it uses `getpass` so the key is not echoed). The `.env` file is ignored by git.
                        - If a key is accidentally exposed, rotate it immediately with the provider.

                        Provider selection and env variable mapping
                        - `LLM_PROVIDER` controls which provider the orchestrator attempts to use (supported: `openai`, `groq`).
                        - Mapped env vars:
                            - `openai` → `OPENAI_API_KEY`
                            - `groq` → `GROQ_API_KEY`

                        The orchestrator will only attempt to initialize CrewAI agents if `ENABLE_LLM=1` and the provider's API key env var is present; otherwise it falls back to local implementations.

                        Packaging for Kaggle
                        --------------------
                        To prepare for Kaggle:

                        1. Ensure `ENABLE_LLM=0` in the notebook environment (Kaggle kernels do not provide external API keys by default).
                        2. Place `hospital_energy.csv` in a Kaggle dataset and attach it to the notebook (accessible at `/kaggle/input/<dataset>/hospital_energy.csv`).
                        3. Update paths for outputs to `/kaggle/working/outputs` if you want the generated images to be downloadable after the run.
                        4. Keep the notebook self-contained: install only necessary packages and avoid network calls.

                        Testing and CI
                        --------------
                        There are basic tests in `tests/` that exercise agent logic and an integration-style workflow test. Recommended CI (GitHub Actions):

                        - Install Python and dependencies
                        - Run `pytest` and also run `python main.py` with `ENABLE_LLM=0` to ensure the non-LLM path works in CI

                        If you'd like, I can add a `.github/workflows/python-tests.yml` workflow that runs on push and PRs.

                        Contributing and license
                        ------------------------
                        - Contributions: open issues or PRs. Keep changes small and include tests.
                        - Code style: follow standard Python style (PEP8). Add type hints where possible.
                        - License: if you plan to publish, add a `LICENSE` file (MIT recommended). I can add one for you.

                        Appendix — file map (key files)

                        - `main.py` — thin runner that starts the orchestrator
                        - `main_workflow.py` — orchestrator, LLM guard and local fallbacks
                        - `agents/` — `data_collector.py`, `analysis_agent.py`, `report_generator.py`, `intervention_planner.py`
                        - `utils/` — `memory.py`, `evaluation.py`
                        - `data/` — `hospital_energy.csv`, `long_term_memory.json`
                        - `outputs/` — produced PNG visualizations (ignored by git by default)
                        - `scripts/save_key.py` — interactive helper to write `.env`
                        - `tests/` — unit and integration tests

                        Contact / next steps
                        --------------------
                        If you want, I can now:

                        1. Add GitHub Actions CI to run tests and the non-LLM run automatically. (Recommended.)
                        2. Build a Kaggle-ready notebook and tune `requirements.txt` for Kaggle compatibility.
                        3. Add a license (MIT) and a CONTRIBUTING guide.

                        Tell me which next step you'd like and I'll implement it and validate the changes locally.
energy_source, emission_factor_kg_co2_per_unit, region
