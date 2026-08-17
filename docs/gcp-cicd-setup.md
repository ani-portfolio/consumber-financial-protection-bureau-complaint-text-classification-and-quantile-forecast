# GCP + GitHub Actions CI/CD Setup Reference

This doc lists the GCP resources and GitHub Actions secrets/variables required to
deploy this project's CI/CD to GCP. It holds **names and purpose only — never
actual credential or resource values**. Nothing described here is provisioned yet
(see CLAUDE.md, "Current State of the Repo"); this is prep reference for when
`.github/workflows/` is built out.

## 1. Identity & auth

A dedicated CI/CD service account (e.g. `github-actions-deployer@<project>.iam.gserviceaccount.com`),
least-privilege, not `Owner`/`Editor`:

| Role | Why |
|---|---|
| `roles/artifactregistry.writer` | push container images |
| `roles/aiplatform.user` | submit Vertex AI PipelineJobs, deploy to Endpoints |
| `roles/storage.objectAdmin` (scoped to specific buckets) | raw data + pipeline artifact buckets |
| `roles/bigquery.dataEditor` + `roles/bigquery.jobUser` (scoped to target dataset) | raw/point-in-time tables, monitoring tables |
| `roles/pubsub.editor` (scoped to `prediction-events`) | publish/subscribe for prediction events |
| `roles/iam.serviceAccountUser` (on itself) | act as the runtime SA for Vertex AI jobs |

**Auth mechanism: Workload Identity Federation (WIF)**, not a downloaded JSON key —
short-lived OIDC tokens for the GitHub Actions runner, no long-lived key material to
leak or rotate. One-time provisioning via `gcloud` (not from this repo):

- Workload Identity Pool
- Workload Identity Provider (OIDC, trusting `token.actions.githubusercontent.com`, scoped to this repo)
- IAM binding granting this GitHub repo permission to impersonate the CI/CD service account

Workflow auth: `google-github-actions/auth` (WIF mode) → `google-github-actions/setup-gcloud`.

## 2. GitHub Actions Secrets / Variables

Populated once WIF + the service account above exist. Set under repo **Settings →
Secrets and Variables → Actions**. Secrets for sensitive values, Variables for
everything else — neither should be hardcoded in workflow YAML.

| Name | Kind | Purpose |
|---|---|---|
| `GCP_PROJECT_ID` | variable | target GCP project |
| `GCP_REGION` | variable | e.g. `us-central1` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | secret | full WIF provider resource name |
| `GCP_SERVICE_ACCOUNT_EMAIL` | variable/secret | CI/CD service account to impersonate |
| `ARTIFACT_REGISTRY_REPO` | variable | container image repo |
| GCS bucket names (raw data, pipeline/model artifacts) | variable | staging + artifact storage |
| BigQuery dataset name(s) | variable | raw + point-in-time base tables |
| Pub/Sub topic name (`prediction-events`) | variable | serving → monitoring event stream |
| Vertex AI resource IDs (Feature Store ID, Endpoint ID, Model Registry location) | variable | set once those resources are created |
| MLflow tracking URI + credentials | secret | only if MLflow is hosted outside GCP |
| `GCP_SA_KEY` | secret | **fallback only** if WIF isn't used — avoid if possible |

## 3. Where things live (don't mix tiers)

1. **Actual secret values** → GitHub repo secrets. Never committed.
2. **Non-secret identifiers** (project ID, region, bucket/dataset/topic names) →
   GitHub repo variables, and mirrored into this repo's typed config objects once
   `configs/` exists (e.g. `configs/gcp.yaml` read via a dataclass, values still
   sourced from env vars at runtime, not hardcoded — see CLAUDE.md guideline #2/#14).
3. **Local developer secrets** → `.env` (already gitignored), prefer
   `gcloud auth application-default login` over static keys where possible.
4. **This doc** — names/purpose only, safe to commit, no credential material.
