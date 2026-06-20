#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from build_fallback_prompts import INSTRUCTION_PROMPT, SYSTEM_PROMPT
from common import PROJECT_ROOT, experiment_test_splits, load_config, read_jsonl, update_manifest, write_jsonl


def load_stage(split: str, strategy: str) -> Dict[str, Dict[str, Any]]:
    path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.verieql_stage.jsonl"
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def build_prompt(sample: Dict[str, Any]) -> str:
    formatted_input = json.dumps(
        {
            "sql1": sample["sql1"],
            "sql2": sample["sql2"],
            "schema": sample.get("schema", {}),
        },
        ensure_ascii=False,
        indent=2,
    )
    return INSTRUCTION_PROMPT + "\n\n" + formatted_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--verieql-strategy", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.verieql_strategy is None:
        args.verieql_strategy = str(cfg.get("experiments", {}).get("autobudget_strategy_name", "autobudget"))

    outputs = {}
    for split in experiment_test_splits(cfg):
        stage = load_stage(split, args.verieql_strategy)
        prompts = []
        for sample in read_jsonl(PROJECT_ROOT / "data/standard" / f"{split}.jsonl"):
            sid = str(sample["id"])
            st = stage.get(sid, {})
            prompts.append(
                {
                    "id": sid,
                    "dataset": split,
                    "gold_label": sample["label"],
                    "verieql_status": st.get("verieql_status", "model_only"),
                    "bucket": st.get("bucket"),
                    "risk_flags": st.get("risk_flags", []),
                    "static_unsupported_flags": st.get("static_unsupported_flags", []),
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": build_prompt(sample),
                }
            )
        out_path = PROJECT_ROOT / "outputs/model_only_prompts" / f"{split}.model_only.prompts.jsonl"
        outputs[split] = {"path": str(out_path), "rows": write_jsonl(out_path, prompts)}
    update_manifest(PROJECT_ROOT / "outputs", "build_model_only_prompts", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
