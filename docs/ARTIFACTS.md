# Artifact Inventory

`paper_artifacts/decisions/` contains the small frozen JSONL files needed for
auditing the paper results:

- `test_calcite_spider.fixed*.verieql_stage.jsonl`: fixed-budget VeriEQL stages.
- `*.qwen25_sft_clean_timeout_llm_verify.final.jsonl`: strict Qwen2.5 profile.
- `*.qwen3_sft_clean_timeout_llm_verify.final.jsonl`: strict Qwen3 profile.
- `*.failure_mode_routed.final.jsonl`: submitted paper-reproduction profile.

The paper-reproduction profile includes 85 `trace_prior` decisions for each SFT
backbone. These are intentionally preserved and explicitly counted by
`src/verify_paper_artifacts.py`.

`paper_artifacts/tables/` contains historical paper tables. Some table JSON
entries reference non-included logs by filename only; the frozen decision files
above are the authoritative auditable artifacts in this repository.

