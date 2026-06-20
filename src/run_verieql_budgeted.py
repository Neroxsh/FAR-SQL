#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, read_jsonl, resolve_path, standard_paths, update_manifest, write_json, write_jsonl
from convert_verieql_ready import convert_schema
from routing_policy import descriptor_from_standard_row, load_policy, select_route
from verieql_simulation import simulate_verieql_record


DETERMINED = {"equivalent", "non_equivalent"}
OFFICIAL_CLI_MAX_CORES = 17


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def run_cli(cfg: Dict[str, Any], input_file: Path, output_file: Path, timeout_sec: int, cores: int, skip_existing: bool) -> Dict[str, Any]:
    if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
        return {"status": "skipped_existing", "input": str(input_file), "output": str(output_file), "timeout_sec": timeout_sec}
    if cores > OFFICIAL_CLI_MAX_CORES:
        return run_cli_outer_sharded(cfg, input_file, output_file, timeout_sec, cores, skip_existing)
    root = Path(cfg["environment"]["verieql_root"])
    vcfg = cfg["verieql"]
    cmd = [
        sys.executable,
        "-m",
        "parallel.cli_within_bound",
        "-f",
        str(input_file),
        "-s",
        str(vcfg["sample_bound"]),
        "--mode",
        str(vcfg["mode"]),
        "--cores",
        str(cores),
        "--integrity_constraint",
        str(vcfg["integrity_constraint"]),
        "-t",
        str(timeout_sec),
        "-o",
        str(output_file),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
    info = {
        "status": "ran" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "timeout_sec": timeout_sec,
        "cores": cores,
        "input": str(input_file),
        "output": str(output_file),
        "wall_time_s": time.time() - t0,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def run_cli_outer_sharded(cfg: Dict[str, Any], input_file: Path, output_file: Path, timeout_sec: int, shards: int, skip_existing: bool) -> Dict[str, Any]:
    """Run VeriEQL with finer outer sharding than the official --cores limit.

    The upstream CLI caps --cores at cpu_count() and assigns one contiguous
    chunk per worker. Some SQL-equivalence workloads have severe long-tail
    chunks, so we split the input round-robin into many small files and run the
    official CLI with --cores 1 per shard. This preserves VeriEQL semantics and
    only changes scheduling granularity.
    """
    if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
        return {"status": "skipped_existing", "input": str(input_file), "output": str(output_file), "timeout_sec": timeout_sec}
    rows = read_jsonl(input_file)
    shards = max(1, min(shards, len(rows)))
    parallelism = max(1, min(int(os.environ.get("NDBC_VERIEQL_OUTER_PARALLEL", OFFICIAL_CLI_MAX_CORES)), shards))
    root = Path(cfg["environment"]["verieql_root"])
    vcfg = cfg["verieql"]
    shard_inputs: List[Path] = []
    shard_outputs: List[Path] = []
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(shards)]
    for idx, row in enumerate(rows):
        shard_rows[idx % shards].append(row)
    for shard_idx, part in enumerate(shard_rows):
        shard_input = output_file.with_name(output_file.name + f".shard{shard_idx:03d}.input.jsonl")
        shard_output = output_file.with_name(output_file.name + f".shard{shard_idx:03d}.jsonl")
        write_jsonl(shard_input, part)
        shard_inputs.append(shard_input)
        shard_outputs.append(shard_output)

    def run_one(shard_idx: int) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            "-m",
            "parallel.cli_within_bound",
            "-f",
            str(shard_inputs[shard_idx]),
            "-s",
            str(vcfg["sample_bound"]),
            "--mode",
            str(vcfg["mode"]),
            "--cores",
            "1",
            "--integrity_constraint",
            str(vcfg["integrity_constraint"]),
            "-t",
            str(timeout_sec),
            "-o",
            str(shard_outputs[shard_idx]),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
        return {
            "shard": shard_idx,
            "returncode": proc.returncode,
            "rows": len(shard_rows[shard_idx]),
            "wall_time_s": time.time() - t0,
            "cmd": cmd,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }

    t0 = time.time()
    shard_infos: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_one, shard_idx) for shard_idx in range(shards)]
        for future in concurrent.futures.as_completed(futures):
            info = future.result()
            shard_infos.append(info)
            if info["returncode"] != 0:
                raise RuntimeError(json.dumps({"status": "failed_shard", **info}, ensure_ascii=False, indent=2))

    merged: List[Dict[str, Any]] = []
    for shard_output in shard_outputs:
        merged.extend(read_jsonl(shard_output))
    merged.sort(key=lambda row: int(row.get("index", 0)))
    if len(merged) != len(rows):
        raise RuntimeError(json.dumps({"status": "merge_count_mismatch", "merged": len(merged), "expected": len(rows), "output": str(output_file)}, ensure_ascii=False))
    write_jsonl(output_file, merged)
    for path in [*shard_inputs, *shard_outputs]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return {
        "status": "ran_outer_sharded",
        "returncode": 0,
        "timeout_sec": timeout_sec,
        "cores": 1,
        "outer_shards": shards,
        "outer_parallelism": parallelism,
        "input": str(input_file),
        "output": str(output_file),
        "wall_time_s": time.time() - t0,
        "shard_wall_time_s": [round(float(info["wall_time_s"]), 3) for info in sorted(shard_infos, key=lambda x: x["shard"])],
    }


def read_raw_results(path: Path, ready_index_to_id: Dict[int, str], timeout_sec: int, sample_bound: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        sid = row.get("id")
        if sid is None and row.get("index") is not None:
            sid = ready_index_to_id.get(int(row["index"]))
        if sid is None:
            continue
        sim = simulate_verieql_record(row, timeout_sec, sample_bound)
        out[str(sid)] = {
            "raw_states": row.get("states", []),
            "raw_err": row.get("err"),
            "normalized_status": sim["normalized_status"],
            "verieql_label": sim["verieql_label"],
            "runtime_s": sim["runtime_s"],
            "attempted_bounds": sim["attempted_bounds"],
            "observed_states": sim["observed_states"],
            "terminal_reason": sim["terminal_reason"],
        }
    return out


def process_split(
    cfg: Dict[str, Any],
    split: str,
    cores: int,
    skip_existing: bool,
    strategy_name: str,
    fixed_budget_sec: Optional[int],
    static_policy: str,
) -> Dict[str, Any]:
    policy: Optional[Dict[str, Any]] = None
    if fixed_budget_sec is None:
        policy_path = resolve_path(cfg.get("routing_policy", {}).get("path", "outputs/budget/routing_policy.json"))
        policy = load_policy(policy_path)
    default_budget = int(cfg["budget"]["default_budget_sec"])
    standard = load_by_id(standard_paths(cfg)[split])

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_by_id: Dict[str, Dict[str, Any]] = {}
    feature_rows: List[Dict[str, Any]] = []
    for sid, std in standard.items():
        descriptor = descriptor_from_standard_row(std)
        features = descriptor["features"]
        bucket = descriptor["bucket"]
        static_flags = descriptor["static_unsupported_flags"]
        risk_flags = descriptor["risk_flags"]
        if fixed_budget_sec is None:
            assert policy is not None
            route_info = select_route(descriptor, policy, default_budget)
        else:
            route_info = {
                "route": "run_verieql",
                "selected_budget": int(fixed_budget_sec),
                "policy_level": "fixed",
                "policy_key": f"fixed{fixed_budget_sec}s",
                "policy_support": None,
                "policy_reference_coverage": None,
                "route_reason": f"fixed_budget={fixed_budget_sec}",
            }
        selected_budget = int(route_info["selected_budget"])
        feature_rows.append(
            {
                "id": sid,
                "dataset": std["dataset"],
                "features": features,
                "bucket": bucket,
                "static_unsupported_flags": static_flags,
                "risk_flags": risk_flags,
                **route_info,
                "extraction_stage": "test_time",
            }
        )
        base = {
            "id": sid,
            "dataset": std["dataset"],
            "gold_label": std["label"],
            "bucket": bucket,
            "features": features,
            "selected_budget": selected_budget,
            "static_unsupported_flags": static_flags,
            "risk_flags": risk_flags,
            "strategy": strategy_name,
            "static_policy": static_policy,
            **route_info,
        }
        if route_info["route"] == "zero_budget":
            stage_by_id[sid] = {
                **base,
                "raw_states": [],
                "raw_err": route_info["route_reason"],
                "normalized_status": "zero_budget_routed",
                "verieql_status": "zero_budget_routed",
                "verieql_label": None,
                "runtime_s": 0.0,
                "verieql_runtime": 0.0,
                "fallback_triggered": True,
                "zero_budget_reason": "learned_zero_budget",
            }
        elif selected_budget <= 0:
            stage_by_id[sid] = {
                **base,
                "raw_states": [],
                "raw_err": "non_positive_budget",
                "normalized_status": "zero_budget_routed",
                "verieql_status": "zero_budget_routed",
                "verieql_label": None,
                "runtime_s": 0.0,
                "verieql_runtime": 0.0,
                "fallback_triggered": True,
                "zero_budget_reason": "non_positive_budget",
            }
        else:
            try:
                schema, constraints = convert_schema(std)
                ready = {
                    "index": len(groups[selected_budget]),
                    "id": sid,
                    "file": split,
                    "pair": [std["sql1"], std["sql2"]],
                    "sql1": std["sql1"],
                    "sql2": std["sql2"],
                    "schema": schema,
                    "constraint": constraints,
                    "semantic equivalence": std["label"] == "yes",
                    "source_dataset": std["dataset"],
                    "source_name": std.get("source_name", std.get("source_path")),
                    "source_index": std.get("source_index"),
                }
                groups[selected_budget].append(ready)
                stage_by_id[sid] = base
            except Exception as exc:
                stage_by_id[sid] = {
                    **base,
                    "raw_states": [],
                    "raw_err": str(exc),
                    "normalized_status": "conversion_error",
                    "verieql_status": "conversion_error",
                    "verieql_label": None,
                    "runtime_s": 0.0,
                    "verieql_runtime": 0.0,
                    "fallback_triggered": True,
                }

    run_infos = []
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    out_dir = PROJECT_ROOT / "outputs/verieql_budgeted"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = PROJECT_ROOT / "outputs/test_time_features" / f"{split}.features.jsonl"
    feature_count = write_jsonl(feature_path, feature_rows)
    for budget, rows in sorted(groups.items()):
        if not rows:
            continue
        group_input = out_dir / f"{split}.{strategy_name}.budget{budget}.verieql_ready.jsonl"
        group_output = out_dir / f"{split}.{strategy_name}.budget{budget}.verieql_output.jsonl"
        write_jsonl(group_input, rows)
        group_cores = max(1, min(cores, len(rows)))
        run_infos.append(run_cli(cfg, group_input, group_output, budget, group_cores, skip_existing))
        ready_index_to_id = {int(row["index"]): row["id"] for row in rows}
        raw_by_id.update(read_raw_results(group_output, ready_index_to_id, budget, int(cfg["verieql"]["sample_bound"])))

    normalized_rows = []
    for sid, std in standard.items():
        stage = stage_by_id[sid]
        if "verieql_status" not in stage:
            raw = raw_by_id.get(sid)
            if raw is None:
                stage = {
                    **stage,
                    "raw_states": [],
                    "raw_err": "missing_verieql_result",
                    "normalized_status": "runtime_error",
                    "verieql_status": "runtime_error",
                    "verieql_label": None,
                    "runtime_s": float(stage.get("selected_budget", 0.0) or 0.0),
                    "verieql_runtime": float(stage.get("selected_budget", 0.0) or 0.0),
                    "fallback_triggered": True,
                }
            else:
                status = raw["normalized_status"]
                stage = {
                    **stage,
                    **raw,
                    "verieql_status": status,
                    "verieql_runtime": float(raw["runtime_s"]),
                    "fallback_triggered": status not in DETERMINED,
                }
        normalized_rows.append(stage)

    normalized_path = out_dir / f"{split}.{strategy_name}.normalized.jsonl"
    decision_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy_name}.verieql_stage.jsonl"
    return {
        "test_time_feature_path": str(feature_path),
        "test_time_feature_rows": feature_count,
        "runs": run_infos,
        "normalized_path": str(normalized_path),
        "normalized_rows": write_jsonl(normalized_path, normalized_rows),
        "decision_path": str(decision_path),
        "decision_rows": write_jsonl(decision_path, normalized_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--strategy-name", default=None)
    ap.add_argument("--fixed-budget-sec", type=int, default=None)
    ap.add_argument(
        "--static-policy",
        choices=["skip", "run"],
        default=None,
        help="skip classifies known unsupported SQL before VeriEQL; run attempts VeriEQL when conversion succeeds.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.strategy_name is None:
        args.strategy_name = str(cfg.get("experiments", {}).get("autobudget_strategy_name", "autobudget"))
    static_policy = args.static_policy or str(cfg["verieql"].get("static_unsupported_policy", "skip"))
    cores = int(os.environ.get("NDBC_VERIEQL_CORES", cfg["verieql"]["cores"]))
    outputs = {}
    datasets = args.datasets or ",".join(experiment_test_splits(cfg))
    for split in [x.strip() for x in datasets.split(",") if x.strip()]:
        outputs[split] = process_split(cfg, split, cores, args.skip_existing, args.strategy_name, args.fixed_budget_sec, static_policy)
    summary_path = PROJECT_ROOT / "outputs/verieql_budgeted/run_summary.json"
    write_json(summary_path, outputs)
    update_manifest(PROJECT_ROOT / "outputs", "run_verieql_budgeted", {"summary": str(summary_path), "outputs": outputs})
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
