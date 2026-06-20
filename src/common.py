#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to read configs/default.yaml. "
            "Please run inside `conda activate EquSQL`."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return cfg


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return p if p.is_absolute() else root / p


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode failed in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(obj)
    return rows


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode failed in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def normalize_label(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "true", "equ", "equivalent", "1"}:
        return "yes"
    if text in {"no", "false", "neq", "inequ", "non_equivalent", "non-equivalent", "0"}:
        return "no"
    return None


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
FALLBACK_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:：]\s*(yes|no)\b", re.I)


def parse_answer(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    tagged_answers = []
    for tag in ANSWER_TAG_RE.finditer(s):
        content = tag.group(1).strip().lower()
        if content in {"yes", "no"}:
            tagged_answers.append(content)
    if tagged_answers:
        return tagged_answers[-1]

    fallback_answers = [m.group(1).lower() for m in FALLBACK_ANSWER_RE.finditer(s)]
    if fallback_answers:
        return fallback_answers[-1]

    stripped = s.strip().lower()
    if stripped in {"yes", "no"}:
        return stripped
    return None


def verieql_candidate_budgets(cfg: Dict[str, Any]) -> List[int]:
    vcfg = cfg.get("verieql", {})
    budgets = vcfg.get("candidate_budgets_sec") or vcfg.get("latency_budgets_sec") or [10, 30, 60, 120]
    return [int(x) for x in budgets]


def verieql_reference_budget(cfg: Dict[str, Any]) -> int:
    vcfg = cfg.get("verieql", {})
    return int(vcfg.get("autobudget_reference_sec") or max(verieql_candidate_budgets(cfg)))


def verieql_official_timeout(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("verieql", {}).get("official_timeout_sec", 600))


def experiment_test_splits(cfg: Dict[str, Any]) -> List[str]:
    return [str(x) for x in cfg.get("experiments", {}).get("test_splits", list(standard_paths(cfg)))]


def stable_id(*parts: Any, prefix: str = "sample") -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return f"{prefix}_{h.hexdigest()[:16]}"


def sql_pair_hash(sql1: str, sql2: str, schema: Any) -> str:
    payload = json.dumps(
        {"sql1": sql1, "sql2": sql2, "schema": schema},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(statistics.mean(vals)) if vals else 0.0


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p)


def update_manifest(outputs_dir: str | Path, step: str, payload: Dict[str, Any]) -> None:
    path = Path(outputs_dir) / "run_manifest.json"
    manifest: Dict[str, Any]
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["steps"].append({"step": step, "time": manifest["updated_at"], **payload})
    write_json(path, manifest)


def standard_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    return {k: resolve_path(v) for k, v in cfg["standard_data"].items()}
