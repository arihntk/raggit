# Golden Datasets for raggit Evaluation

This directory contains **committed golden datasets** – curated ground-truth for every evaluation tier, used to test the system and as baselines for regression.

## Files

| File | Tier | Description |
|------|------|-------------|
| `golden.yaml` / `golden-all-tiers.yaml` | `all` | All tiers: 14 component types + ingestion/retrieval/e2e pipelines + system (53 metrics, 27+13 tests) – **use this to test the entire system** |
| `golden-system.yaml` | `system` | Comprehensive system (12 tests, 22 metrics covering retrieval, tenant/tag/prefix filters, safety, citations, latency, MCP) |
| `golden-component-*.yaml` | `component` | Per-primitive isolated suites (parser 3 tests, chunker 3, cleaner 3, pii 3, injection 3, sanitizer 2, embedder 2, etc.) |
| `golden-pipeline-*.yaml` | `pipeline` | Ingestion and retrieval chains |

## Usage

```bash
# Run the built-in golden
uv run raggit eval eval_datasets/golden.yaml
uv run raggit eval eval_datasets/golden-component-chunker.yaml
uv run raggit eval eval_datasets/golden-system.yaml --output report.json

# Create your own golden dataset
uv run raggit eval --generate --kind system --name my-golden
# Edit my-golden.yaml with your queries, expected_chunk_ids, expected_answer, filters

# Merge your custom golden with the built-in
uv run raggit eval my-custom.yaml --golden-dataset eval_datasets/golden-system.yaml

# Use your custom golden as sole dataset
uv run raggit eval --golden-dataset ./my-golden.yaml

# Compare against a previous report (Δ column)
uv run raggit eval eval_datasets/golden-component-cleaner.yaml --output report.json
uv run raggit eval eval_datasets/golden-component-cleaner.yaml --golden-report report.json
```

## Custom Golden Dataset Format

A custom golden dataset is just an `EvalDataset` YAML/JSON with `kind`, `metrics`, and tests. Example:

```yaml
name: my-team-golden
kind: system
metrics: [retrieval_recall@k, answer_contains, filter_tenant_accuracy]
k_values: [5, 10]
tests:
  - id: team-query-1
    query: "What is our SLA?"
    expected_chunk_ids: ["11111111-1111-1111-1111-111111111111"]
    expected_answer: "Our SLA is 99.9% uptime"
    filters: {tenant_id: "my-tenant"}
    tags: [team, sla]
```

See `raggit eval --list-metrics` for 69+ available metrics and `raggit eval --generate --kind component --component <name> --help` for templates.

## Adding to CI

```bash
uv run raggit eval eval_datasets/golden-all-tiers.yaml --output report.json
# Fail CI if any test fails (exit code 1)
```

The terminal report renders a detailed per-tier breakdown (header, aggregates with Δ, per-test table, component/pipeline details, answer preview, verdict) – the same rich output is available via `--output report.md`.
