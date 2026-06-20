#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, parse_answer, read_jsonl, update_manifest, write_jsonl


def build_llm(model_path: str, cfg: Dict[str, Any]):
    from vllm import LLM

    fallback_cfg = cfg["fallback"]
    return LLM(
        model=model_path,
        tensor_parallel_size=int(fallback_cfg.get("tensor_parallel_size", 1)),
        max_model_len=int(fallback_cfg.get("max_model_len", 16000)),
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        gpu_memory_utilization=float(fallback_cfg.get("gpu_memory_utilization", 0.9)),
    )


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_fallback_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    fallback_cfg = dict(cfg["fallback"])
    if args.num_samples is not None:
        fallback_cfg["num_samples"] = args.num_samples
    if args.temperature is not None:
        fallback_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        fallback_cfg["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        fallback_cfg["max_new_tokens"] = args.max_new_tokens
    return fallback_cfg


def run_generation(
    llm: Any,
    prompts: List[Dict[str, Any]],
    fallback_cfg: Dict[str, Any],
    limit: Optional[int],
    split: str,
    model_key: str,
) -> List[Dict[str, Any]]:
    from vllm import SamplingParams

    if limit:
        prompts = prompts[:limit]
    sampling_params = SamplingParams(
        n=int(fallback_cfg["num_samples"]),
        temperature=float(fallback_cfg["temperature"]),
        top_p=float(fallback_cfg["top_p"]),
        max_tokens=int(fallback_cfg["max_new_tokens"]),
        stop=["<|im_end|>", "</s>"],
    )
    results = []
    batch_size = int(os.environ.get("FALLBACK_BATCH_SIZE", fallback_cfg.get("batch_size", 900)))
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        print(
            json.dumps(
                {
                    "event": "fallback_batch_start",
                    "model_key": model_key,
                    "split": split,
                    "start": start,
                    "end": start + len(batch),
                    "total": len(prompts),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        messages_list = [
            [
                {"role": "system", "content": item.get("system_prompt", "You are a helpful SQL expert.")},
                {"role": "user", "content": item["prompt"]},
            ]
            for item in batch
        ]
        t0 = time.time()
        outputs = llm.chat(messages_list, sampling_params)
        elapsed = time.time() - t0
        per_item_runtime = elapsed / len(batch) if batch else 0.0
        for item, output in zip(batch, outputs):
            raw_outputs = [candidate.text.strip() for candidate in output.outputs]
            parsed = [parse_answer(text) for text in raw_outputs]
            results.append(
                {
                    **{k: item[k] for k in ["id", "dataset", "gold_label", "verieql_status", "bucket", "risk_flags", "static_unsupported_flags"]},
                    "num_samples": int(fallback_cfg["num_samples"]),
                    "decoding": {
                        "backend": "vllm.chat",
                        "temperature": float(fallback_cfg["temperature"]),
                        "top_p": float(fallback_cfg["top_p"]),
                    },
                    "raw_outputs": raw_outputs,
                    "parsed_answers": parsed,
                    "has_parse_failed": any(x not in {"yes", "no"} for x in parsed),
                    "fallback_runtime": per_item_runtime,
                    "prompt_variant": item.get("prompt_variant"),
                }
            )
        print(
            json.dumps(
                {
                    "event": "fallback_batch_done",
                    "model_key": model_key,
                    "split": split,
                    "start": start,
                    "end": start + len(batch),
                    "elapsed_s": elapsed,
                    "per_item_runtime_s": per_item_runtime,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--model-key", default=None, help="Run only one configured fallback model key.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-dir", default="outputs/fallback_prompts")
    ap.add_argument("--prompt-suffix", default=None)
    ap.add_argument("--log-dir", default="outputs/fallback_logs")
    ap.add_argument("--log-suffix", default="fallback")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.strategy is None:
        args.strategy = str(cfg.get("experiments", {}).get("autobudget_strategy_name", "autobudget"))
    fallback_cfg = resolve_fallback_cfg(cfg, args)
    models = [m for m in cfg["fallback"].get("models", []) if m.get("enabled", True)]
    if args.model_key:
        models = [m for m in models if m.get("key") == args.model_key]
    if not models:
        raise ValueError("No enabled fallback models matched the requested selection.")
    outputs = {}
    for model_spec in models:
        model_key = str(model_spec["key"])
        model_path = str(model_spec["path"])
        print(json.dumps({"event": "fallback_model_load_start", "model_key": model_key, "model_path": model_path}, ensure_ascii=False), flush=True)
        llm = build_llm(model_path, cfg)
        try:
            for split in experiment_test_splits(cfg):
                prompt_name = f"{split}.{args.strategy}.prompts.jsonl" if not args.prompt_suffix else f"{split}.{args.strategy}.{args.prompt_suffix}.prompts.jsonl"
                prompt_path = PROJECT_ROOT / args.prompt_dir / prompt_name
                prompts = read_jsonl(prompt_path)
                logs = run_generation(llm, prompts, fallback_cfg, args.limit, split, model_key)
                for row in logs:
                    row["model_key"] = model_key
                    row["model_path"] = model_path
                out_path = PROJECT_ROOT / args.log_dir / f"{split}.{args.strategy}.{model_key}.{args.log_suffix}.jsonl"
                outputs[f"{split}:{model_key}"] = {"path": str(out_path), "rows": write_jsonl(out_path, logs)}
        finally:
            del llm
            cleanup_gpu()
    update_manifest(PROJECT_ROOT / "outputs", "run_fallback_gpu", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
