#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import PROJECT_ROOT, load_config, read_jsonl, update_manifest, verieql_candidate_budgets, verieql_reference_budget, write_json
from routing_policy import learn_policy, write_policy


def policy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rcfg = cfg.get("routing_policy", {})
    return {
        "path": rcfg.get("path", "outputs/budget/routing_policy.json"),
        "candidate_budgets_sec": rcfg.get("candidate_budgets_sec"),
        "reference_budget_sec": rcfg.get("reference_budget_sec"),
        "zero_budget_levels": rcfg.get("zero_budget_levels", ["fine", "medium", "compat_bucket"]),
        "min_support": rcfg.get("min_support", {"fine": 80, "medium": 120, "bucket": 1}),
    }


def rel_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def group_rows_for_report(policy: Dict[str, Any], level_name: str, budget_filter: int | None = None, limit: int = 20) -> List[Dict[str, Any]]:
    level = next((x for x in policy.get("levels", []) if x.get("name") == level_name), None)
    if not level:
        return []
    rows = list(level.get("groups", {}).values())
    if budget_filter is not None:
        rows = [row for row in rows if int(row.get("selected_budget", -1)) == budget_filter]
    rows.sort(key=lambda row: (int(row.get("support", 0)), float(row.get("reference_coverage", 0.0))), reverse=True)
    return rows[:limit]


def write_markdown(path: Path, policy: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Learned Routing Policy\n\n")
        f.write("This is the learned non-parametric routing policy used by AutoBudget. It maps SQL feature signatures to an action in `{0s, 1s, 2s, 3s, 5s, 10s}`.\n\n")
        f.write(f"Version: `{policy['version']}`\n\n")
        f.write(f"Budgets: `{policy['budgets_sec']}`\n\n")
        f.write(f"Reference budget: `{policy['reference_budget_sec']}`\n\n")
        f.write(f"Max policy budget: `{policy['max_policy_budget_sec']}`\n\n")
        f.write(f"Epsilon: `{policy['epsilon']}`\n\n")
        f.write(f"Min coverage threshold: `{policy['min_coverage_threshold']}`\n\n")
        f.write(f"Zero-budget levels: `{policy.get('zero_budget_levels', [])}`\n\n")
        f.write("Static unsupported indicators are policy features, not separate hard-coded exceptions.\n\n")
        f.write("## Bucket Backoff Table\n\n")
        f.write("| Bucket | Selected Budget |\n")
        f.write("|---|---:|\n")
        for bucket, budget in sorted(policy.get("bucket_budget_table", {}).items()):
            f.write(f"| {bucket} | {budget} |\n")
        f.write("\n## Policy Level Sizes\n\n")
        f.write("| Level | Min Support | Groups |\n")
        f.write("|---|---:|---:|\n")
        for level in policy.get("levels", []):
            f.write(f"| {level['name']} | {level['min_support']} | {len(level.get('groups', {}))} |\n")

        for title, level_name, budget_filter in [
            ("Largest Fine Groups", "fine", None),
            ("Fine Zero-Budget Action Groups", "fine", 0),
            ("Medium Zero-Budget Action Groups", "medium", 0),
            ("Compatibility Zero-Budget Action Groups", "compat_bucket", 0),
        ]:
            rows = group_rows_for_report(policy, level_name, budget_filter=budget_filter)
            f.write(f"\n## {title}\n\n")
            f.write("| Key | Support | Selected | Ref. Coverage | Det@1s | Det@2s | Det@120s |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                cov = row.get("coverage_by_budget", {})
                f.write(
                    f"| `{row['key']}` | {row['support']} | {row['selected_budget']} | "
                    f"{row['reference_coverage']:.2%} | {cov.get('1', 0.0):.2%} | "
                    f"{cov.get('2', 0.0):.2%} | {cov.get('120', 0.0):.2%} |\n"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    rcfg = policy_config(cfg)
    profile_path = PROJECT_ROOT / "outputs/calibration/calibration_verieql_profile.jsonl"
    rows = read_jsonl(profile_path)
    budgets = [int(x) for x in (rcfg["candidate_budgets_sec"] or verieql_candidate_budgets(cfg))]
    reference_budget = int(rcfg["reference_budget_sec"] or verieql_reference_budget(cfg))
    policy = learn_policy(
        rows,
        budgets=budgets,
        epsilon=float(cfg["budget"]["epsilon"]),
        min_coverage=float(cfg["budget"]["min_coverage_threshold"]),
        default_budget=int(cfg["budget"]["default_budget_sec"]),
        min_support_by_level={str(k): int(v) for k, v in rcfg["min_support"].items()},
        reference_budget=reference_budget,
        zero_budget_levels=[str(x) for x in rcfg.get("zero_budget_levels", ["fine", "medium", "compat_bucket"])],
    )
    out_path = rel_path(rcfg["path"])
    write_policy(out_path, policy)
    write_json(PROJECT_ROOT / "outputs/budget/budget_table.json", policy["bucket_budget_table"])
    report_path = PROJECT_ROOT / "outputs/tables/routing_policy_report.md"
    write_markdown(report_path, policy)
    payload = {
        "policy_path": str(out_path),
        "report_path": str(report_path),
        "profile_rows": len(rows),
        "bucket_budget_table": policy["bucket_budget_table"],
        "level_group_counts": {level["name"]: len(level.get("groups", {})) for level in policy.get("levels", [])},
    }
    update_manifest(PROJECT_ROOT / "outputs", "learn_routing_policy", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
