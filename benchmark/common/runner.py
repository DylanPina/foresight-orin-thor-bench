"""Coordinator/worker orchestration shared by every runtime.

Each model runs in its own worker process so that the runtime's CUDA state and
memory are released between models. ``run_benchmark`` prepares the shared
inputs, launches ``python -m <module> --worker-model ... --config ...`` per
model, collects the per-model result JSON, and writes the reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import validate_batch_settings
from .utils import (
    device_metadata,
    prepare_images,
    reclaim_jetson_memory,
    slug,
    workspace_root,
    write_json,
    write_reports,
)

RunModel = Callable[[str, dict[str, Any]], dict[str, Any]]
BuildConfig = Callable[[argparse.Namespace, list[dict[str, Any]], Path], dict[str, Any]]


def run_worker(model: str, config_path: Path, run_model: RunModel) -> int:
    """Run one model and persist a result for the coordinating process."""
    config_path = config_path.expanduser().resolve()
    result_path = config_path.parent / "models" / f"{slug(model)}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = run_model(model, config)
    except Exception as exc:
        formatted_traceback = traceback.format_exc()
        print(formatted_traceback, file=sys.stderr, flush=True)
        result = {
            "model": model,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": formatted_traceback,
        }
    write_json(result_path, result)
    return 0 if result.get("status") == "ok" else 1


def default_output_dir(runtime_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return workspace_root() / "data" / "benchmarks" / runtime_dir / timestamp


def run_benchmark(
    args: argparse.Namespace,
    *,
    worker_module: str,
    runtime: str,
    runtime_dir: str,
    report_title: str,
    build_config: BuildConfig,
) -> int:
    """Prepare inputs and coordinate the per-model worker loop."""
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(runtime_dir)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_image_count = validate_batch_settings(
        args.batches, args.images_per_batch, len(args.images)
    )
    selected_sources = args.images[:selected_image_count]
    try:
        images = prepare_images(selected_sources, output_dir / "images")
    except Exception as exc:
        print(f"Failed to prepare benchmark images: {exc}", file=sys.stderr)
        return 2

    config = build_config(args, images, output_dir)
    config_path = output_dir / "run_config.json"
    write_json(config_path, config)

    results: list[dict[str, Any]] = []
    for model in args.models:
        print(f"\n=== Benchmarking {model} ===", flush=True)
        result_path = output_dir / "models" / f"{slug(model)}.json"
        log_path = output_dir / "models" / f"{slug(model)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pre_reclaim = reclaim_jetson_memory(args.reclaim_memory)
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    worker_module,
                    "--worker-model",
                    model,
                    "--config",
                    str(config_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=workspace_root(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_stream.write(line)
            returncode = process.wait()
        post_reclaim = reclaim_jetson_memory(args.reclaim_memory)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "model": model,
                "status": "error",
                "error": f"worker exited {returncode} without a result file",
            }
        result["worker_log"] = str(log_path)
        result["memory_reclamation"] = {
            "before_worker": pre_reclaim,
            "after_worker": post_reclaim,
        }
        write_json(result_path, result)
        results.append(result)
        print(f"{model}: {result['status']}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "device": device_metadata(),
        "config": config,
        "images": images,
        "models": results,
    }
    write_reports(output_dir, payload, title=report_title)
    print(f"\nResults written to {output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(result["status"] == "ok" for result in results) else 1
