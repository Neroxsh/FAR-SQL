# Reproducibility

## Artifact Audit

The easiest reproducibility check uses frozen artifacts only:

```bash
PYTHONPATH=src python src/verify_paper_artifacts.py
```

This recomputes:

- Fixed-budget VeriEQL under the submitted monotone cumulative protocol.
- `strict-farsql` results.
- `paper-reproduction` results, including the number of prior-only decisions.

## Full Pipeline

Full reruns require external assets that are not committed:

- VeriEQL and its converter.
- Raw calibration and test JSONL files.
- Local SFT checkpoints.
- A GPU environment for vLLM.

Put raw data under `data/raw/`, model checkpoints under `models/`, and VeriEQL
under `third_party/`, or edit `configs/default.yaml`.

The CPU/formal stages are:

```bash
bash scripts/00_preflight.sh
bash scripts/00_prepare_data.sh
bash scripts/01_calibrate_budget.sh
bash scripts/01b_build_calibration_audit_and_routing.sh
bash scripts/02_run_verieql_budgeted_test.sh
bash scripts/10_run_verieql_fixed_budgets.sh
```

The GPU/model stages are:

```bash
bash scripts/03_build_fallback_prompts.sh
bash scripts/04_run_fallback_gpu.sh
bash scripts/05_fuse_decisions.sh
bash scripts/09_compute_all_experiments.sh
```

To regenerate strict FAR-SQL decisions after model logs exist:

```bash
PYTHONPATH=src python src/run_status_aware_router.py \
  --strategy adaptive_accuracy \
  --model-key qwen3_1_7b_100sft_cot \
  --timeout-positive-prior-mode llm_verify \
  --output-tag strict_farsql_qwen3
```

To regenerate the submitted paper-reproduction profile, opt into the prior
explicitly:

```bash
PYTHONPATH=src python src/run_status_aware_router.py \
  --strategy adaptive_accuracy \
  --model-key qwen3_1_7b_100sft_cot \
  --timeout-positive-prior-mode direct_yes \
  --output-tag paper_reproduction_qwen3
```

## Metric Conventions

`Acc_eq` and `Acc_neq` keep all samples in the denominator. `GM` is the
geometric mean of those two class accuracies. `Und` counts outputs that are not
valid yes/no labels.

For fixed VeriEQL, the submitted paper uses a monotone cumulative protocol: a
longer budget preserves any determined label already found by a shorter budget.
The P95 latency reported by the artifact audit uses lower-rank P95, matching
the submitted tables.
