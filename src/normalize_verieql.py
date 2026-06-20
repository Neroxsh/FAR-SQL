#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PROJECT_ROOT, load_config, read_jsonl, standard_paths, update_manifest, verieql_reference_budget, write_jsonl
from verieql_simulation import elapsed_for_state

REFUTED_STATES = {"NEQ", "SAT", "REFUTED"}
CHECKED_STATES = {"EQU", "UNSAT", "CHECKED", "VERIFIED"}
TIMEOUT_STATES = {"TMO"}
UNSUPPORTED_STATES = {"NSE", "NIE"}
CONVERSION_STATES = {"SYN"}
RUNTIME_ERROR_STATES = {"OOM", "OTE"}


def flatten_times(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)):
        if not math.isnan(float(value)):
            out.append(float(value))
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_times(item))
    return out


def state_elapsed(value: Any, state: str, default_timeout: float) -> float:
    return elapsed_for_state(value, state, default_timeout)


def raw_runtime(row: Dict[str, Any], status: str, default_timeout: float) -> float:
    states = [str(x).upper() for x in row.get("states", []) if x is not None]
    times_by_state = row.get("times")
    if isinstance(times_by_state, list):
        elapsed = 0.0
        terminal_by_status = {
            "non_equivalent": {"NEQ"},
            "timeout": {"TMO"},
            "unsupported_runtime": {"NSE", "NIE"},
            "conversion_error": {"SYN"},
            "unknown": {"UNK"},
            "runtime_error": {"OOM", "OTE"},
        }
        if status == "equivalent":
            for i, state in enumerate(states):
                if i >= len(times_by_state):
                    break
                elapsed += state_elapsed(times_by_state[i], state, default_timeout)
            return elapsed
        targets = terminal_by_status.get(status, set())
        for i, state in enumerate(states):
            if i >= len(times_by_state):
                break
            elapsed += state_elapsed(times_by_state[i], state, default_timeout)
            if state in targets:
                return elapsed
        if elapsed > 0:
            return elapsed
    times = flatten_times(row.get("times"))
    if times and status not in {"timeout", "unsupported_runtime"}:
        return sum(times)
    err = str(row.get("err", ""))
    if "Time Out" in err:
        return default_timeout
    return 0.0


def classify_for_leetcode_dataset(row: Dict[str, Any]) -> Optional[str]:
    """Match KDD/leetcode/spilt-equ-inequ.py exactly for dataset construction."""
    states = [str(x).strip().upper() for x in row.get("states", []) if x is not None and str(x).strip()]
    if any(x in REFUTED_STATES for x in states):
        return "NEQ"
    if any(x in CHECKED_STATES for x in states):
        return "EQU"
    return None


def _has_clean_err(err: str) -> bool:
    return not err or err.strip() in {"NA", "None", "null"}


def normalize_status(row: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Normalize one VeriEQL record for experiment-time decisions.

    This is intentionally stricter than the LeetCode dataset construction
    script. For example, states like ["EQU", "EQU", "TMO"] are treated as
    timeout, because VeriEQL did not finish a final safe decision.
    """
    states = [str(x).upper() for x in row.get("states", []) if x is not None]
    err = str(row.get("err", "") or "")
    low = err.lower()
    if "not equivalent" in low or any(x in REFUTED_STATES for x in states) or row.get("counterexample"):
        return "non_equivalent", "no"
    if "time out" in low or "timeout" in low or any(x in TIMEOUT_STATES for x in states):
        return "timeout", None
    if "not supported feature" in low or "not implemented" in low or "not supported" in low:
        return "unsupported_runtime", None
    if any(x in UNSUPPORTED_STATES for x in states):
        return "unsupported_runtime", None
    if "unknowncolumn" in low or "unknowndatabase" in low:
        return "conversion_error", None
    if "syntaxerror" in low or "parsersyntaxerror" in low or "SYN" in states:
        return "conversion_error", None
    if "unknown" in low or "undecidable" in low or "unk" in states:
        return "unknown", None
    if any(x in RUNTIME_ERROR_STATES for x in states) or "exception" in low or "traceback" in low:
        return "runtime_error", None
    if any(x in CHECKED_STATES for x in states) and _has_clean_err(err):
        return "equivalent", "yes"
    if not _has_clean_err(err):
        return "runtime_error", None
    return "unknown", None


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def process_dataset(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    max_budget = verieql_reference_budget(cfg)
    standard = load_by_id(standard_paths(cfg)[name])
    features = load_by_id(PROJECT_ROOT / "outputs/features" / f"{name}.features.jsonl")

    source_spec = cfg.get("data_sources", {}).get(name, {})
    uses_precomputed = source_spec.get("format") == "verieql_record"
    raw_path = (
        Path(source_spec["path"])
        if uses_precomputed
        else PROJECT_ROOT / "outputs/verieql_maxrun" / f"{name}.bound{cfg['verieql']['sample_bound']}.jsonl"
    )
    ready_index_to_id: Dict[int, str] = {}
    if not uses_precomputed:
        ready_rows = read_jsonl(PROJECT_ROOT / "outputs/verieql_ready" / f"{name}.verieql_ready.jsonl")
        ready_index_to_id = {int(row["index"]): row["id"] for row in ready_rows}
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    if raw_path.exists():
        for source_index, row in enumerate(read_jsonl(raw_path), 1):
            sid = row.get("id")
            if sid is None and uses_precomputed:
                sid = f"{name}:{source_index}"
            if sid is None and row.get("index") is not None:
                sid = ready_index_to_id.get(int(row["index"]))
            if sid is not None:
                raw_by_id[str(sid)] = row

    conversion_errors = (
        {}
        if uses_precomputed
        else load_by_id(PROJECT_ROOT / "outputs/verieql_ready" / f"{name}.conversion_errors.jsonl")
    )
    normalized = []
    for sid, std in standard.items():
        feat = features.get(sid, {})
        static_flags = feat.get("static_unsupported_flags", [])
        if static_flags:
            status, label, runtime = "unsupported_static", None, 0.0
            raw_states, raw_err = [], ";".join(static_flags)
        elif sid in conversion_errors:
            status, label, runtime = "conversion_error", None, 0.0
            raw_states, raw_err = [], conversion_errors[sid].get("error")
        else:
            raw = raw_by_id.get(sid)
            if raw is None:
                status, label, runtime = "runtime_error", None, 0.0
                raw_states, raw_err = [], "missing_verieql_result"
            else:
                status, label = normalize_status(raw)
                runtime = raw_runtime(raw, status, float(max_budget))
                raw_states, raw_err = raw.get("states", []), raw.get("err")
        normalized.append(
            {
                "id": sid,
                "dataset": std["dataset"],
                "gold_label": std["label"],
                "bucket": feat.get("bucket", "Unknown"),
                "raw_states": raw_states,
                "raw_err": raw_err,
                "normalized_status": status,
                "verieql_label": label,
                "runtime_s": runtime,
                "static_unsupported_flags": static_flags,
                "risk_flags": feat.get("risk_flags", []),
            }
        )
    out_path = PROJECT_ROOT / "outputs/verieql_maxrun" / f"{name}.normalized.jsonl"
    return {"path": str(out_path), "rows": write_jsonl(out_path, normalized)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--datasets", default=None, help="Comma-separated dataset names. Defaults to all standard datasets.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = list(standard_paths(cfg))
    if args.datasets:
        names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    outputs = {name: process_dataset(cfg, name) for name in names}
    update_manifest(PROJECT_ROOT / "outputs", "normalize_verieql", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
