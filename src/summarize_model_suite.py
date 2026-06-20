#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from common import PROJECT_ROOT, load_config, parse_answer, read_jsonl, write_json


def majority_label(row: Dict[str, Any]) -> Optional[str]:
    parsed = [x for x in row.get("parsed_answers", []) if x in {"yes", "no"}]
    if not parsed:
        return None
    counts = Counter(parsed)
    if counts["yes"] == counts["no"]:
        return None
    return "yes" if counts["yes"] > counts["no"] else "no"


def metric_block(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    eq_rows = [r for r in rows if r.get("gold_label") == "yes"]
    neq_rows = [r for r in rows if r.get("gold_label") == "no"]

    def acc(subset: List[Dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        correct = sum(1 for r in subset if r.get("pred_label") == r.get("gold_label"))
        return correct / len(subset)

    eq_acc = acc(eq_rows)
    neq_acc = acc(neq_rows)
    overall = acc(rows)
    gm = math.sqrt(eq_acc * neq_acc) if eq_rows and neq_rows else 0.0
    return {
        "equivalent_acc": eq_acc,
        "non_equivalent_acc": neq_acc,
        "gm": gm,
        "overall_acc": overall,
        "samples": len(rows),
    }


def as_percent(x: float) -> float:
    return round(x * 100.0, 2)


def load_model_only(split: str, model_key: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl"
    rows = read_jsonl(path)
    return [{"gold_label": r["gold_label"], "pred_label": majority_label(r)} for r in rows]


def load_fallback(split: str, strategy: str, model_key: str, suffix: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{strategy}.{model_key}.{suffix}.jsonl"
    rows = read_jsonl(path)
    return [{"gold_label": r["gold_label"], "pred_label": majority_label(r)} for r in rows]


def load_final(split: str, strategy: str, model_key: str, output_tag: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.{model_key}.{output_tag}.final.jsonl"
    rows = read_jsonl(path)
    return [{"gold_label": r["gold_label"], "pred_label": r.get("final_label")} for r in rows]


def collect_rows(model_key: str, strategy: str, router_base_tag: str) -> List[Dict[str, Any]]:
    methods = [
        ("纯模型（三次投票）", "model_only", None),
        ("动态预算 + Trace引导", "fallback", "fallback_trace_guided"),
        ("动态预算 + Witness引导", "fallback", "fallback_witness_guided"),
        ("动态预算 + VeriEQL极简弱提示", "fallback", "fallback_verieql_hint"),
        ("动态预算 + 状态感知融合（干净版）", "final", f"{router_base_tag}_clean"),
        ("动态预算 + 状态感知融合（增强版）", "final", f"{router_base_tag}_fine"),
    ]
    all_rows: List[Dict[str, Any]] = []
    for display_name, method_type, tag in methods:
        row: Dict[str, Any] = {"方法": display_name}
        for split in ["test_leetcode", "test_calcite_spider"]:
            if method_type == "model_only":
                data = load_model_only(split, model_key)
            elif method_type == "fallback":
                data = load_fallback(split, strategy, model_key, str(tag))
            else:
                data = load_final(split, strategy, model_key, str(tag))
            metrics = metric_block(data)
            prefix = "LeetCode" if split == "test_leetcode" else "Calcite+Spider"
            row[f"{prefix}_等价Acc"] = as_percent(metrics["equivalent_acc"])
            row[f"{prefix}_不等价Acc"] = as_percent(metrics["non_equivalent_acc"])
            row[f"{prefix}_GM"] = as_percent(metrics["gm"])
            row[f"{prefix}_总体Acc"] = as_percent(metrics["overall_acc"])
        all_rows.append(row)
    return all_rows


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "方法",
        "LeetCode_等价Acc",
        "LeetCode_不等价Acc",
        "LeetCode_GM",
        "LeetCode_总体Acc",
        "Calcite+Spider_等价Acc",
        "Calcite+Spider_不等价Acc",
        "Calcite+Spider_GM",
        "Calcite+Spider_总体Acc",
    ]
    lines = []
    lines.append("| " + " | ".join(h.replace("_", " ") for h in headers) + " |")
    lines.append("|" + "|".join(["---"] + [":---:" for _ in headers[1:]]) + "|")
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--strategy", default="adaptive_accuracy")
    ap.add_argument("--router-base-tag", default="status_aware_router_base")
    ap.add_argument("--output-prefix", default=None)
    args = ap.parse_args()
    _cfg = load_config(args.config)

    rows = collect_rows(args.model_key, args.strategy, args.router_base_tag)
    prefix = args.output_prefix or args.model_key
    csv_path = PROJECT_ROOT / "outputs/tables" / f"{prefix}.suite_summary.csv"
    md_path = PROJECT_ROOT / "outputs/tables" / f"{prefix}.suite_summary.md"
    json_path = PROJECT_ROOT / "outputs/metrics" / f"{prefix}.suite_summary.json"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, rows)
    write_json(json_path, {"model_key": args.model_key, "rows": rows})
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
