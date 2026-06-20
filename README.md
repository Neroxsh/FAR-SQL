# FAR-SQL

FAR-SQL is a failure-aware hierarchical pipeline for SQL equivalence judgment.
It keeps determined VeriEQL outputs as formal decisions and routes only
non-determined cases to local SQL-equivalence SFT models.

This repository contains the code, standardized Calcite+Spider-DAIL test data,
and frozen artifacts used to audit the submitted paper results.

## Result Profiles

Two profiles are intentionally kept.

| Profile | Meaning |
|---|---|
| `strict-farsql` | The literal method description: VeriEQL `equivalent`/`non_equivalent` decisions are preserved; every non-determined state goes to the local model with Self-Consistency@3. |
| `paper-reproduction` | The submitted-result reproduction profile. It preserves the same formal-first boundary, but also includes a timeout-positive-prior heuristic for 85 Calcite+Spider-DAIL samples. The verifier prior is reported explicitly as `Prior-only` by the audit script. |

Run the artifact audit:

```bash
PYTHONPATH=src python src/verify_paper_artifacts.py
```

Expected headline rows:

```text
paper-reproduction  Qwen2.5-SFT  GM 83.98  Prior-only 85
paper-reproduction  Qwen3-SFT    GM 83.69  Prior-only 85
strict-farsql       Qwen2.5-SFT  GM 80.14  Prior-only 0
strict-farsql       Qwen3-SFT    GM 80.53  Prior-only 0
```

## Repository Layout

```text
configs/              Runtime and experiment configuration.
data/standard/        Standardized test JSONL files.
paper_artifacts/      Frozen decision files and paper tables.
scripts/              End-to-end experiment entry points.
src/                  Data preparation, routing, fusion, evaluation code.
tests/                Small regression tests for artifact metrics.
```

Generated outputs should go under `outputs/`; frozen submitted artifacts stay
under `paper_artifacts/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python src/verify_paper_artifacts.py
```

The audit path above does not require GPUs, model checkpoints, or VeriEQL.

## Full Reproduction

A full rerun requires:

- VeriEQL and its SQL-format converter.
- Raw LeetCode/Calcite/Spider-DAIL input files.
- The local SFT checkpoints for `qwen25_coder_1_5b_100sft_cot` and
  `qwen3_1_7b_100sft_cot`.
- A vLLM-compatible GPU environment for model generation.

Edit `configs/default.yaml` or create your own config with local paths, then run:

```bash
bash scripts/00_preflight.sh
bash scripts/00_prepare_data.sh
bash scripts/01_calibrate_budget.sh
bash scripts/01b_build_calibration_audit_and_routing.sh
bash scripts/02_run_verieql_budgeted_test.sh
bash scripts/10_run_verieql_fixed_budgets.sh
bash scripts/03_build_fallback_prompts.sh
bash scripts/04_run_fallback_gpu.sh
bash scripts/05_fuse_decisions.sh
bash scripts/09_compute_all_experiments.sh
```

See `docs/REPRODUCIBILITY.md` for profile-specific notes.

## Citation

If this artifact is useful, please cite the accompanying paper:

```bibtex
@misc{farsql2026,
  title = {Failure-Aware Routing for SQL Equivalence Judgment},
  author = {Xing Shihao},
  year = {2026},
  note = {FAR-SQL artifact}
}
```
