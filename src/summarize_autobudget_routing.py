#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import PROJECT_ROOT, experiment_test_splits, load_config, read_jsonl, resolve_path, standard_paths, update_manifest, write_json, write_jsonl
from routing_policy import load_policy, route_rows, summarize_routes


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AutoBudget Routing Plan\n\n")
        f.write("This table is computed without running VeriEQL. It applies the learned routing policy to the test sets and shows which samples receive a zero-budget action or a symbolic timeout action.\n\n")
        f.write(f"Policy: `{payload['policy_path']}`\n\n")
        for split, item in payload["splits"].items():
            summary = item["summary"]
            f.write(f"## {split}\n\n")
            f.write(f"Samples: `{summary['samples']}`\n\n")
            f.write(f"Route counts: `{summary['by_route']}`\n\n")
            f.write(f"Budget counts: `{summary['by_budget']}`\n\n")
            f.write(f"Policy level counts: `{summary['by_policy_level']}`\n\n")
            f.write(f"Static flag counts: `{summary['static_flag_counts']}`\n\n")
            f.write(f"Zero-budget samples with static flags: `{summary['zero_budget_with_static_flags']}`\n\n")
            f.write("| Bucket | zero_budget | run_verieql |\n")
            f.write("|---|---:|---:|\n")
            for bucket, counts in summary["by_bucket_route"].items():
                f.write(
                    f"| {bucket} | {counts.get('zero_budget', 0)} | {counts.get('run_verieql', 0)} |\n"
                )
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    policy_path = resolve_path(cfg.get("routing_policy", {}).get("path", "outputs/budget/routing_policy.json"))
    policy = load_policy(policy_path)
    paths = standard_paths(cfg)
    payload: Dict[str, Any] = {"policy_path": str(policy_path), "bucket_budget_table": policy.get("bucket_budget_table", {}), "splits": {}}
    for split in experiment_test_splits(cfg):
        routed = route_rows(
            read_jsonl(paths[split]),
            policy,
            default_budget=int(cfg["budget"]["default_budget_sec"]),
        )
        detail_path = PROJECT_ROOT / "outputs/budget" / f"{split}.autobudget_routing.jsonl"
        row_count = write_jsonl(detail_path, routed)
        payload["splits"][split] = {
            "detail_path": str(detail_path),
            "rows": row_count,
            "summary": summarize_routes(routed),
        }
    write_json(PROJECT_ROOT / "outputs/budget/autobudget_routing_plan.json", payload)
    write_markdown(PROJECT_ROOT / "outputs/tables/autobudget_routing_plan.md", payload)
    update_manifest(PROJECT_ROOT / "outputs", "summarize_autobudget_routing", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
