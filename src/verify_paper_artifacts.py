#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
YES_NO = {"yes", "no"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    # Match the submitted paper tables: lower-rank percentile rather than
    # interpolation. For 412 samples, P95 uses the 391st zero-based item.
    index = max(0, min(len(vals) - 1, math.floor((pct / 100.0) * len(vals)) - 1))
    return vals[index]


def metrics_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    yes_rows = [r for r in rows if r.get("gold_label") == "yes"]
    no_rows = [r for r in rows if r.get("gold_label") == "no"]
    yes_correct = sum(1 for r in yes_rows if r.get("final_label") == "yes")
    no_correct = sum(1 for r in no_rows if r.get("final_label") == "no")
    n = len(rows)
    acc_eq = 100.0 * yes_correct / len(yes_rows) if yes_rows else 0.0
    acc_neq = 100.0 * no_correct / len(no_rows) if no_rows else 0.0
    return {
        "n": n,
        "acc_eq": acc_eq,
        "acc_neq": acc_neq,
        "gm": math.sqrt(acc_eq * acc_neq),
        "overall": 100.0 * (yes_correct + no_correct) / n if n else 0.0,
        "und": sum(1 for r in rows if r.get("final_label") not in YES_NO),
    }


def summarize_decision_sources(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(r.get("decision_source")) for r in rows)
    counts["non_llm_prior"] = counts.get("trace_prior", 0)
    return dict(counts)


def stage_as_decisions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("verieql_status")
        if status == "equivalent":
            final_label = "yes"
        elif status == "non_equivalent":
            final_label = "no"
        else:
            final_label = "uncertain"
        out.append(
            {
                "gold_label": row.get("gold_label"),
                "final_label": final_label,
                "total_runtime": float(row.get("verieql_runtime", 0.0) or 0.0),
                "decision_source": "verified" if final_label in YES_NO else "abstained",
            }
        )
    return out


def fixed_monotone_rows(stage_by_budget: Dict[int, List[Dict[str, Any]]], budget: int) -> List[Dict[str, Any]]:
    """Apply the paper's monotone cumulative VeriEQL protocol up to `budget`.

    If a shorter budget already produced a determined VeriEQL label, the longer
    budget keeps that label. Later budgets can only fill previously unresolved
    samples. Cross-budget equivalent/inequivalent conflicts are treated as an
    error because the paper audit found none.
    """
    decided: Dict[str, Dict[str, Any]] = {}
    for current_budget in sorted(b for b in stage_by_budget if b <= budget):
        for row in stage_by_budget[current_budget]:
            sid = str(row["id"])
            status = row.get("verieql_status")
            label = "yes" if status == "equivalent" else "no" if status == "non_equivalent" else "uncertain"
            if label not in YES_NO:
                decided.setdefault(
                    sid,
                    {
                        "gold_label": row.get("gold_label"),
                        "final_label": "uncertain",
                        "decision_source": "abstained",
                    },
                )
                continue
            previous = decided.get(sid)
            if previous and previous.get("final_label") in YES_NO:
                if previous["final_label"] != label:
                    raise ValueError(f"Cross-budget VeriEQL conflict for {sid}: {previous['final_label']} vs {label}")
                continue
            decided[sid] = {
                "gold_label": row.get("gold_label"),
                "final_label": label,
                "decision_source": "verified",
            }
    return list(decided.values())


def runtime_summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    runtimes = [float(r.get("total_runtime", r.get("verieql_runtime", 0.0)) or 0.0) for r in rows]
    return {
        "avg_s": sum(runtimes) / len(runtimes) if runtimes else 0.0,
        "p95_s": percentile(runtimes, 95),
    }


def summarize_file(profile: str, backbone: str, path: Path, stage: bool = False) -> Dict[str, Any]:
    raw_rows = read_jsonl(path)
    rows = stage_as_decisions(raw_rows) if stage else raw_rows
    metrics = metrics_from_rows(rows)
    return {
        "profile": profile,
        "backbone": backbone,
        "file": str(path.relative_to(ROOT)),
        **metrics,
        **runtime_summary(rows),
        "decision_sources": summarize_decision_sources(rows),
    }


def collect(root: Path) -> List[Dict[str, Any]]:
    d = root / "paper_artifacts" / "decisions"
    fixed_budgets = [10, 30, 60, 120]
    stage_by_budget = {
        budget: read_jsonl(d / f"test_calcite_spider.fixed{budget}s.verieql_stage.jsonl") for budget in fixed_budgets
    }
    rows = []
    for budget in fixed_budgets:
        direct_rows = stage_as_decisions(stage_by_budget[budget])
        monotone_rows = fixed_monotone_rows(stage_by_budget, budget)
        metrics = metrics_from_rows(monotone_rows)
        rows.append(
            {
                "profile": "verieql-fixed-monotone",
                "backbone": f"{budget}s",
                "file": str((d / f"test_calcite_spider.fixed{budget}s.verieql_stage.jsonl").relative_to(ROOT)),
                **metrics,
                **runtime_summary(direct_rows),
                "decision_sources": summarize_decision_sources(monotone_rows),
            }
        )

    specs = [
        (
            "strict-farsql",
            "Qwen2.5-SFT",
            d / "test_calcite_spider.adaptive_accuracy.qwen25_coder_1_5b_100sft_cot.qwen25_sft_clean_timeout_llm_verify.final.jsonl",
            False,
        ),
        (
            "strict-farsql",
            "Qwen3-SFT",
            d / "test_calcite_spider.adaptive_accuracy.qwen3_1_7b_100sft_cot.qwen3_sft_clean_timeout_llm_verify.final.jsonl",
            False,
        ),
        (
            "paper-reproduction",
            "Qwen2.5-SFT",
            d / "test_calcite_spider.adaptive_accuracy.qwen25_coder_1_5b_100sft_cot.failure_mode_routed.final.jsonl",
            False,
        ),
        (
            "paper-reproduction",
            "Qwen3-SFT",
            d / "test_calcite_spider.adaptive_accuracy.qwen3_1_7b_100sft_cot.failure_mode_routed.final.jsonl",
            False,
        ),
    ]
    rows.extend(summarize_file(profile, backbone, path, stage=stage) for profile, backbone, path, stage in specs)
    return rows


def format_row(row: Dict[str, Any]) -> str:
    sources = row["decision_sources"]
    prior = sources.get("non_llm_prior", 0)
    return (
        f"| {row['profile']} | {row['backbone']} | {row['acc_eq']:.2f} | {row['acc_neq']:.2f} | "
        f"{row['gm']:.2f} | {row['overall']:.2f} | {row['und']} | {row['avg_s']:.2f}/{row['p95_s']:.2f} | {prior} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of markdown.")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    rows = collect(root)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print("| Profile | Backbone/Budget | Acc_eq | Acc_neq | GM | Overall | Und | Avg/P95 s | Prior-only |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
