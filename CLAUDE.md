# CFPB Hierarchical Financial Text Classification & Forecasting

## Project Mission
An end-to-end ML system that processes, classifies, and forecasts Consumer Financial Protection Bureau (CFPB) complaint data.

1. **Hierarchical Text Classifier (primary)** — predict a complaint's `Product` (Level 1) and `Sub-product` (Level 2) from its free-text narrative. Core challenges: severe class imbalance, multi-task hierarchical label space, and temporal taxonomy drift.
2. **Weekly Complaint Volume Quantile Forecaster (secondary)** — forecast weekly complaint volume per product category as a quantile forecast (p10/p50/p90) from the ingested pipeline data.

**Data source**: CFPB Consumer Complaint Database (public, no API key). Bulk download: `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`. The live API supports weekly incremental ingestion; the bulk CSV is the fastest path for local experimentation.

---

## Current State of the Repo

This project is early-stage — the architecture below is the **target design**, not yet built. Don't assume any component exists until you've checked. As of now:

- Not a git repository yet.
- `src/` contains only a package `__init__.py` — no ingestion, transformation, model, forecasting, evaluation, pipeline, serving, or monitoring modules exist yet.
- `experimentation/` is currently empty — no notebooks yet.
- `pyproject.toml` — pinned deps (pandas/sklearn/torch/transformers/lightgbm/fastapi under `[project.dependencies]`; pytest/ruff/mypy under the `dev` extra).
- No `tests/`, `Dockerfile`, `configs/`, `.github/workflows/`, or `docs/rfcs/` yet.
- No GCP resources (BigQuery, Vertex AI Pipelines, Vertex AI Endpoint, Vertex AI Model Registry, Vertex AI Feature Store, Pub/Sub, GCS) provisioned — the architecture in the next section describes where this is headed, not what's live.

When asked to build a piece of this system, check what already exists in `src/` first rather than assuming the full pipeline is wired up.

---

## Target Architecture (GCP-centric)

```mermaid
flowchart TD
    A["<b>CFPB Public API</b>"]

    subgraph sgIngestion [Ingestion]
        direction TB
        B["<b>Vertex AI Pipeline: Ingest</b><br/>Kubeflow/KFP component; weekly PipelineJob schedule<br/>• Pull CFPB API → stage raw JSON (GCS) → load BigQuery raw table<br/>• Transform (BigQuery SQL) → point-in-time base table: Rolling-Origin Split (never random) · Features Knowable at Prediction Time · DQ &amp; Null Audits"]
        C["<b>Cloud Storage (Raw JSON)</b><br/>raw JSON staged per pull"]
        D["<b>BigQuery (Raw Table — Append-Only)</b>"]
        E["<b>BigQuery (Point-in-Time Base Table)</b>"]
    end

    subgraph sgTraining [Training]
        direction TB
        F["<b>Vertex AI Pipeline</b><br/>authored with Kubeflow/KFP SDK<br/>• Extract → Validate → Feature Build<br/>• Train (Baseline / DistilBERT Multi-Task / PyTorch Quantile Forecaster)<br/>• Evaluate (Pinball Loss/Quantile + Calibration; Macro/Per-Class F1)<br/>• Conditional Register: Champion/Challenger Gate"]
        G["<b>Vertex AI Feature Store</b><br/>online store; entity-level features (classifier)<br/>+ time-series covariates (forecaster)"]
        H["<b>MLflow Experiments</b><br/>Loss/F1 by Segment, Calibration,<br/>Data Snapshot Version"]
        I["<b>Vertex AI Model Registry</b><br/>registers only if it beats champion"]
    end

    S["<b>Client / Caller</b><br/>consumer-facing system or internal<br/>service invoking the API"]

    subgraph sgServing [Serving]
        direction TB
        K["<b>FastAPI</b><br/>custom container on Vertex AI Endpoint<br/>• /v1/predict, /health (Pydantic-validated)<br/>• Feature Store lookup, then run inference<br/>• Async publish request + resolved features + prediction to Pub/Sub (every call)<br/>• Rollout: Shadow → Canary/Traffic-Split → Full<br/>• Fallback: analytical rule if endpoint unavailable"]
    end

    subgraph sgMonitoring [Monitoring &amp; Retraining]
        direction TB
        L["<b>Pub/Sub Topic</b><br/>prediction-events<br/>carries request, resolved features &amp; prediction"]
        P["<b>Reference Distribution</b><br/>baseline feature distribution from<br/>the training-time snapshot"]
        Q["<b>Current Distribution</b><br/>live feature &amp; prediction distribution<br/>from the audit table"]
        M["<b>BigQuery Monitoring Jobs</b><br/>• PSI per feature, per segment (scheduled)<br/>• Rolling pinball loss / F1 &amp; quantile coverage"]
        R["<b>PSI / Quality Metrics Table</b><br/>results keyed by feature, segment, date"]
        N["<b>Drift &amp; Quality Dashboard</b><br/>rendering surface: not yet decided"]
        O["<b>Vertex AI Pipeline: Monitor &amp; Trigger</b><br/>• Own recurring PipelineJob schedule (same pattern as Ingest — paced to data freshness, not just breach signal)<br/>• Reads PSI / rolling-metric tables<br/>• Conditional: breach? → submit Train/Eval PipelineJob"]
    end

    A --> B
    B --> C
    B --> D
    D -->|Transform via BigQuery SQL, same PipelineJob as Ingest| E
    E --> F
    E -->|training snapshot| P
    F -->|feature build| G
    G -->|online lookup by entity ID| K
    F -->|conditional register| H
    F -->|conditional register| I
    I -->|deploys model artifact| K
    S -->|POST /v1/predict| K
    K -->|prediction response| S
    K --> L
    L -->|BigQuery subscription| Q
    P --> M
    Q --> M
    M -->|writes results| R
    R --> N
    R --> O
    O -->|submits training run on breach| F

    classDef external fill:#eceff1,stroke:#455a64,stroke-width:1.5px,color:#000
    classDef pipeline fill:#bbdefb,stroke:#1565c0,stroke-width:1.5px,color:#000
    classDef artifact fill:#c8e6c9,stroke:#2e7d32,stroke-width:1.5px,color:#000
    classDef messaging fill:#ffe0b2,stroke:#ef6c00,stroke-width:1.5px,color:#000
    classDef serving fill:#e1bee7,stroke:#6a1b9a,stroke-width:1.5px,color:#000
    classDef monitoring fill:#ffcdd2,stroke:#c62828,stroke-width:1.5px,color:#000

    style sgIngestion fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 4 3
    style sgTraining fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 4 3
    style sgServing fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 4 3
    style sgMonitoring fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 4 3

    class A,S external
    class B,F,O pipeline
    class C,D,E,G,H,I,P,Q,R artifact
    class L messaging
    class K serving
    class M,N monitoring
```

Gray = external source · Blue = Vertex AI Pipelines · Green = data stores/artifacts · Orange = messaging · Purple = serving · Red = monitoring

### Component notes

- **Framing & RFC** — before building, write a short design doc: target definition, baseline, offline metric (pinball loss per quantile + calibration coverage for the forecaster; macro/per-class F1 for the classifier), online/business metric, latency & volume requirements, label source, and leakage risks. Circulate to Data Engineering, product, and backend before implementation starts.
- **Ingestion (`src/ingestion/`)** — the first component of the Vertex AI Pipeline (authored with the Kubeflow/KFP SDK), triggered by Vertex AI Pipelines' own recurring `PipelineJob` schedule (no standalone Cloud Run job). Pulls the previous 7 days of `date_received` records from the CFPB API, stages raw JSON in GCS (`gs://bucket/raw/YYYY/MM/DD/`), and loads it into an append-only BigQuery raw table for point-in-time reconstruction. Transformation (below) is chained on as this same pipeline's next stage.
- **Transformation (`src/transformation/`)** — BigQuery SQL builds a point-in-time-correct base table: only features knowable at prediction time. Rolling-origin backtest, **never a random split** — hard requirement to avoid temporal leakage. Plus DQ audits (row-count anomalies, narrative completeness, schema validation, missingness). Runs as the final stage of Ingest's own `PipelineJob`, immediately after the raw-table load — not a separate schedule or separate pipeline, so there's no window where new raw data has landed but the base table hasn't been rebuilt yet.
- **Models (`src/models/`, `src/forecasting/`)** — TF-IDF + Logistic Regression/Linear SVM baseline and `distilbert-base-uncased` (PyTorch, HF Transformers) with a joint multi-task head for the Product/Sub-product classifier; PyTorch for the weekly complaint-volume quantile forecaster (P10/P50/P90, pinball loss), with HuggingFace pretrained components used elsewhere where they help. Every run tracked in MLflow: loss/F1 by segment, calibration/coverage, and the data snapshot version. Forecaster output is translated into a downstream decision policy and simulated against history to quantify its impact before shipping.
- **Pipelines (`src/pipelines/`)** — authored with Kubeflow (KFP SDK), executed as Vertex AI Pipelines. Components, each its own container: `Extract (BigQuery) -> Validate -> Feature Build -> Train -> Evaluate -> Conditional Register`. Feature Build also batch-ingests the computed features into Vertex AI Feature Store's online store, so serving reads the same features training was built on. The register step is the champion/challenger gate — it writes to the Vertex AI Model Registry only if the challenger beats the current champion on the offline metric.
- **Evaluation (`src/evaluation/`)** — pinball loss per quantile + calibration coverage for the forecaster; per-class F1, confusion matrices, sliced metrics on rare/tail sub-products for the classifier (not just global accuracy/macro-F1). A coverage breach is the early-warning signal in production.
- **Feature Store** — Vertex AI Feature Store's online store, populated by the training pipeline's Feature Build step (the same computation that builds the BigQuery point-in-time base table), so training and serving read features from one shared computation path and never skew apart. Holds entity-level engineered features (consumer/company complaint history, rolling aggregate counts) for the classifier and time-series covariates (lag/rolling-volume features by product category) for the forecaster — the raw narrative itself still goes straight to DistilBERT, unstored.
- **Serving (`src/serving/`)** — FastAPI, containerized, deployed to a Vertex AI Endpoint (no Cloud Run). `GET /health`, `POST /v1/predict` (Pydantic-validated request/response, returns predicted hierarchy + confidence). On each request: an online Feature Store lookup by entity ID, then the model runs inference directly. Every request — including the resolved feature values from that Feature Store lookup, not just the raw input — plus the prediction is published asynchronously to a Pub/Sub topic (`prediction-events`), off the request's critical path, and delivered into the BigQuery audit table via a native Pub/Sub → BigQuery subscription (no Dataflow). The ML function owns correctness of what runs in the container; Data Engineering owns endpoint reliability.
- **Rollout** — shadow mode first (model scores live traffic, no decision uses it) → a small percentage of live decisions → full traffic, via canary/traffic-split on the Vertex AI Endpoint with rollback on breach. Fallback to the existing analytical rule if the endpoint is unavailable.
- **CI/CD (GitHub Actions, `.github/workflows/`)** — PR: `ruff`/`black`, `mypy`, `pytest`, feature-transform tests, a training smoke test on sampled data. Merge to main: build & push the container image to Artifact Registry, run the full pipeline in staging, run the evaluation gate, deploy to a staging Vertex AI Endpoint, integration tests. Promotion: manual approval, then canary/traffic-split on the endpoint with rollback on breach. Model artifact is versioned separately from code.
- **Monitoring & Retraining (`src/monitoring/`)** — the ML function owns model quality, Data Engineering owns operational health. Drift: BigQuery scheduled queries compute PSI per input feature/segment and category-mix drift by comparing two explicit tables — the **Reference Distribution** (the training-time feature snapshot the current champion was trained on, sourced from the point-in-time base table) against the **Current Distribution** (live feature and prediction values from the Pub/Sub-fed audit table) — which is what makes per-feature PSI possible at all rather than just prediction/output-level drift. This detection job can run more frequently than retraining (e.g. daily) purely for early visibility, since it's cheap and commits no training compute. Both the drift and quality jobs write their computed results — PSI/category-mix values, rolling pinball loss/F1, quantile coverage — into a separate **PSI / Quality Metrics Table**, keyed by feature/segment/date, distinct from the Reference/Current Distribution tables they read from; without that separate table there'd be no time series to threshold against, just a single ephemeral query result each run. Quality: a scheduled BigQuery job joins past predictions to realized outcomes, tracking rolling pinball loss/F1 and quantile coverage. A dedicated **Monitor & Trigger pipeline** — itself a Vertex AI Pipeline on its own recurring `PipelineJob` schedule, deliberately the same pattern as Ingestion rather than faster — reads that metrics table and, on a breach, submits a new run of the training pipeline via the Vertex AI Pipelines SDK. That pacing is intentional, not incidental: submitting a retrain before new data has landed would just refit the existing training snapshot, produce essentially the same model, and can't beat the champion at the Conditional Register gate — wasted compute for no possible upside, so the trigger check is paced to data freshness, not just to the breach signal. This is in addition to, not instead of, the existing CI-triggered (on merge to `main`) and manual/ad hoc trigger paths — all three converge on the same pipeline and evaluation gate, so a bad retrain never auto-promotes regardless of what triggered it. Dashboard rendering surface isn't decided yet.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/                # ci.yml, cd.yml — lint/unit/feature-transform/training-smoke-test on PR;
│                                  #   build+push image, staging pipeline run, eval gate, staging endpoint
│                                  #   deploy, integration tests on merge; manual-approval promotion with
│                                  #   canary/rollback — not created yet
├── .gitignore                    # .env, credentials, __pycache__, .venv — not created yet
├── configs/                      # data/model/pipeline yaml configs — not created yet
├── docs/
│   └── rfcs/                     # design docs from the Framing & RFC step — not created yet
├── experimentation/               # exploratory notebooks only — currently empty
├── src/
│   ├── __init__.py
│   ├── ingestion/                 # first component of the Vertex AI Pipeline: CFPB pull, GCS/BigQuery load — not created yet
│   ├── transformation/            # BigQuery SQL: point-in-time base table, rolling-origin split, DQ audits;
│   │                               #   chained as Ingest's pipeline's final stage, not its own pipeline — not created yet
│   ├── models/                    # TF-IDF/SVM baseline + DistilBERT multi-task classifier — not created yet
│   ├── forecasting/                # PyTorch quantile volume forecaster (P10/P50/P90) — not created yet
│   ├── evaluation/                 # MLflow/Vertex AI experiment tracking, sliced metrics — not created yet
│   ├── pipelines/                  # Kubeflow/KFP SDK pipelines: (1) Extract→Validate→Feature Build (incl. batch
│   │                               #   ingest to Vertex AI Feature Store)→Train→Evaluate→Conditional Register,
│   │                               #   (2) Monitor & Trigger (reads drift/quality tables, submits (1) on breach)
│   │                               #   — not created yet
│   ├── serving/                    # FastAPI on Vertex AI Endpoint (/predict, /health): Feature Store online-lookup
│   │                               #   client + Pub/Sub prediction-event publisher — not created yet
│   └── monitoring/                 # PSI/drift + rolling quality monitoring, Looker Studio queries — not created yet
├── tests/
│   ├── unit/                       # test_*.py mirroring src/, mocked GCP clients, conftest.py — not created yet
│   └── integration/                # CI-only, staging project, conftest.py — not created yet
├── pyproject.toml
└── CLAUDE.md
```

---

## Development Commands

```bash
# Activate the existing venv (Python 3.11) before installing or running anything
source .venv/bin/activate

# Install/refresh dependencies
pip install -e ".[dev]"

# Download the bulk CFPB dataset locally (large — several GB)
python -c "from src.ingestion.load_data import download_bulk_dataset; download_bulk_dataset()"

# Lint / type-check / test (deps are in pyproject.toml; tool configs not yet added)
ruff check .
mypy src
pytest

# Unit tests only (fast, mocked GCP clients, no cloud calls)
pytest tests/unit

# Integration tests (hit a staging project — CI-only, not for local runs)
pytest tests/integration
```

---

## Shorthand Commands

- **"commit and push"** — stage the relevant changes, create a commit with a very concise message (one short line, no elaboration), and push to the current branch's remote. Skip the usual confirmation step for this exact phrase; still follow all other git safety rules (no `--force`, no `--no-verify`, no amending, never commit files that look like they contain secrets).

## Development Guidelines

1. **Deterministic processing** — always split by `date_received`, never randomly; this is required to avoid temporal leakage in both the classifier and the forecaster.
2. **Type safety** — Pydantic models for API payloads, dataclasses/typed configs for pipeline settings.
3. **Notebooks are exploratory only** — code in `experimentation/` must be refactored into `src/` modules with unit tests before it's considered done; don't leave production logic living only in a notebook.
4. **Always use the venv** — `source .venv/bin/activate` before installing packages or running scripts.
5. **Cost-conscious infra** — there's no scale-to-zero serving layer in this design; tune the Vertex AI Endpoint's min-replica count and pipeline component machine types for cost, and prefer scheduled/batch execution over persistent compute wherever serving isn't required.
6. **Don't assume unbuilt infra** — GCP resources (BigQuery, Vertex AI Pipelines, Vertex AI Endpoint, Vertex AI Model Registry, Vertex AI Feature Store, Pub/Sub, GCS) described in the target architecture are not provisioned. Building against them requires setting them up first; check before assuming a table, bucket, endpoint, feature store, or topic exists.
7. **All plots use seaborn** — every chart in notebooks (`experimentation/`) or `src/` must be produced with `seaborn` plotting functions (`sns.barplot`, `sns.lineplot`, `sns.histplot`, etc.), not raw `matplotlib` calls or pandas' `.plot()`. `matplotlib.pyplot` is still fine for figure-level setup (`plt.figure`, `plt.title`, `plt.xlabel`, `plt.show`) — it's the mark-drawing call itself that must go through seaborn.
8. **RFC before major model/architecture changes** — circulate a short design doc (target definition, baseline, offline/online metric, latency & volume, label source & leakage) to Data Engineering, product, and backend before starting classifier/forecaster work.
9. **Champion/challenger discipline** — never register or promote a model outside the Vertex AI Pipeline's conditional-register gate; all retraining, including ad hoc runs and automatic drift-triggered runs from the Monitor & Trigger pipeline, goes through the same pipeline and evaluation gate so a bad retrain never auto-promotes. An automatic trigger gets no special promotion privileges — it's just another way to start the same gated pipeline.
10. **Rollout discipline** — new models go live via shadow mode (scoring live traffic, no decision uses it), then a partial traffic/decision split (canary), then full traffic, with rollback on breach and a defined fallback if the serving endpoint is unavailable.
11. **Ownership split** — the ML function owns model quality and the serving container's correctness; Data Engineering owns endpoint/operational reliability. Keep this boundary in mind when scoping monitoring and on-call work.
12. **Structured logging** — use Python's `logging` module everywhere, never `print()`. Pipeline components and the serving container emit structured/JSON logs so they're queryable in Cloud Logging: INFO for normal flow (request received, Feature Store lookup, pipeline stage start/end), WARNING for degraded-but-recovered paths (fallback triggered, Pub/Sub publish retried), ERROR for failures needing attention. Never log raw complaint narratives, resolved feature values, or other PII — log a hash/ID instead; full narrative text and the resolved feature vector used for a prediction only ever land in the access-controlled BigQuery audit table, reached via the async Pub/Sub → BigQuery subscription (guideline on Serving above), not application logs.
13. **pytest conventions** — test files named `test_*.py`, mirroring the `src/` module they cover, under `tests/unit/` or `tests/integration/`. Unit tests mock external GCP clients (BigQuery, GCS, Vertex AI) and never hit real cloud resources; only integration tests, run in CI against a staging project, are allowed to. Use `pytest.mark.parametrize` for edge cases (empty narrative, unseen taxonomy label, tail sub-products) instead of near-duplicate test functions. Shared fixtures live in `conftest.py` per test tier.
14. **Secrets & config** — never hardcode credentials, project IDs, bucket names, or dataset names inline; source them from the typed config objects (guideline #2) and Secret Manager for anything sensitive. Credential files and `.env` are gitignored, never committed.
