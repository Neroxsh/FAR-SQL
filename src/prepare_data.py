#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    PROJECT_ROOT,
    load_config,
    normalize_label,
    parse_json_maybe,
    read_jsonl,
    standard_paths,
    update_manifest,
    write_json,
    write_jsonl,
)


def get_input_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = parse_json_maybe(row.get("input", {}))
    if isinstance(raw, dict):
        return raw
    return {}


def standardize_test_row(
    row: Dict[str, Any],
    dataset: str,
    source_path: str,
    default_label: Optional[str],
    source_index: int,
) -> Optional[Dict[str, Any]]:
    sql1 = row.get("sql1")
    sql2 = row.get("sql2")
    if not isinstance(sql1, str) or not isinstance(sql2, str):
        inner = get_input_obj(row)
        sql1 = inner.get("sql1")
        sql2 = inner.get("sql2")
    if not isinstance(sql1, str) or not isinstance(sql2, str):
        return None

    schema = row.get("schema")
    if schema is None:
        schema = get_input_obj(row).get("schema", {})
    constraint = row.get("constraint")

    label = normalize_label(row.get("semantic equivalence"))
    if label is None:
        label = normalize_label(row.get("label"))
    if label is None:
        label = normalize_label(default_label)
    if label is None:
        return None

    raw_id = row.get("id", row.get("index", source_index))
    sid = f"{dataset}:{Path(source_path).stem}:{raw_id}"
    return {
        "id": sid,
        "dataset": dataset,
        "schema": parse_json_maybe(schema),
        "constraint": parse_json_maybe(constraint) if constraint is not None else None,
        "sql1": sql1,
        "sql2": sql2,
        "label": label,
        "source_name": Path(source_path).name,
        "source_index": source_index,
        "source_difficulty": row.get("difficulty"),
    }


def limit_rows(rows: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def build_test_split(cfg: Dict[str, Any], split_name: str, output_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    spec = cfg["data_sources"][split_name]
    rows: List[Dict[str, Any]] = []
    for source in spec["sources"]:
        source_path = source["path"]
        part = []
        for i, row in enumerate(read_jsonl(source_path), 1):
            std = standardize_test_row(row, spec["dataset"], source_path, source.get("label"), i)
            if std is not None:
                part.append(std)
        rows.extend(limit_rows(part, limit))
    write_jsonl(output_path, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="Limit rows per source for smoke tests.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = standard_paths(cfg)

    leetcode = build_test_split(cfg, "test_leetcode", paths["test_leetcode"], args.limit)
    calcite_spider = build_test_split(cfg, "test_calcite_spider", paths["test_calcite_spider"], args.limit)

    manifest = {
        "test_leetcode": {"path": str(paths["test_leetcode"]), "rows": len(leetcode)},
        "test_calcite_spider": {"path": str(paths["test_calcite_spider"]), "rows": len(calcite_spider)},
        "note": "Only final test inputs are standardized here. Calibration reads the precomputed VeriEQL record directly.",
    }
    write_json(PROJECT_ROOT / "data/standard/dataset_manifest.json", manifest)
    update_manifest(PROJECT_ROOT / "outputs", "prepare_data", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
