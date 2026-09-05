"""vLLM model execution and benchmark orchestration loops."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import (
    ResourceMonitor,
    device_metadata,
    package_version,
    prepare_images,
    reclaim_jetson_memory,
    slug,
    summarize_runs,
    workspace_root,
    write_json,
    write_reports,
)
from .config import (
    DATASET_SIZE,
    build_run_config,
    parse_args,
    validate_batch_settings,
)


def _build_prompt_and_images(
    processor: Any, image_paths: list[str], prompt_text: str
) -> tuple[str, list[Any]]:
    from PIL import Image

    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    content = [{"type": "image", "image": path} for path in image_paths]
    content.append({"type": "text", "text": prompt_text})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt, images


def _run_request(llm: Any, request: dict[str, Any], sampling_params: Any) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        outputs = llm.generate(request, sampling_params=sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        elapsed = time.perf_counter() - started
        resources = monitor.stop(elapsed)

    output = outputs[0]
    completion = output.outputs[0]
    metrics = output.metrics
    output_tokens = len(completion.token_ids)
    ttft_seconds = getattr(metrics, "first_token_latency", None) if metrics else None
    first_token_ts = getattr(metrics, "first_token_ts", 0.0) if metrics else 0.0
    last_token_ts = getattr(metrics, "last_token_ts", 0.0) if metrics else 0.0
    decode_seconds = (
        last_token_ts - first_token_ts
        if last_token_ts and first_token_ts and last_token_ts >= first_token_ts
        else None
    )
    decode_token_count = max(output_tokens - 1, 0)
    torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    torch_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    if torch_peak_allocated_mb == 0 and torch_peak_reserved_mb == 0:
        torch_peak_allocated_mb = None
        torch_peak_reserved_mb = None
        allocator_warning = (
            "Torch allocator peaks are unavailable because vLLM owns CUDA memory "
            "in its EngineCore subprocess"
        )
        resources["warning"] = "; ".join(
            warning for warning in (resources.get("warning"), allocator_warning) if warning
        )
    return {
        "ttft_ms": ttft_seconds * 1000.0 if ttft_seconds is not None else None,
        "e2e_latency_ms": elapsed * 1000.0,
        "output_tokens": output_tokens,
        "output_throughput_tokens_s": output_tokens / elapsed if elapsed > 0 else None,
        "decode_throughput_tokens_s": (
            decode_token_count / decode_seconds
            if decode_seconds and decode_token_count
            else None
        ),
        "tpot_ms": (
            decode_seconds * 1000.0 / decode_token_count
            if decode_seconds is not None and decode_token_count
            else None
        ),
        "finish_reason": completion.finish_reason,
        "generated_text": completion.text,
        "torch_peak_allocated_mb": torch_peak_allocated_mb,
        "torch_peak_reserved_mb": torch_peak_reserved_mb,
        **resources,
    }


def run_model(model: str, config: dict[str, Any]) -> dict[str, Any]:
    """Load and benchmark one model inside a worker process."""
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch inside the container")
    image_paths = [item["path"] for item in config["images"]]
    batch_count = int(config["batches"])
    images_per_batch = int(config["images_per_batch"])
    expected_image_count = validate_batch_settings(
        batch_count,
        images_per_batch,
        int(config.get("dataset_size", DATASET_SIZE)),
    )
    if len(image_paths) != expected_image_count:
        raise ValueError(
            f"configuration requires {expected_image_count} prepared images, "
            f"got {len(image_paths)}"
        )
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    requests = []
    images = []
    for start in range(0, len(image_paths), images_per_batch):
        batch_paths = image_paths[start : start + images_per_batch]
        prompt, batch_images = _build_prompt_and_images(
            processor, batch_paths, config["prompt"]
        )
        images.extend(batch_images)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": batch_images}})
    engine_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "max_model_len": config["max_model_len"],
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": images_per_batch, "video": 0, "audio": 0},
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "disable_log_stats": False,
        "seed": 0,
        "enforce_eager": False,
    }
    if config.get("kv_cache_memory_bytes") is not None:
        engine_kwargs["kv_cache_memory_bytes"] = config["kv_cache_memory_bytes"]
    if model.lower().startswith("google/gemma-4"):
        engine_kwargs["mm_processor_kwargs"] = {"max_soft_tokens": 280}
        engine_kwargs["attention_config"] = {"backend": "TRITON_ATTN"}

    llm = None
    try:
        load_started = time.perf_counter()
        llm = LLM(**engine_kwargs)
        load_seconds = time.perf_counter() - load_started
        structured_output = bool(config.get("structured_output", False))
        structured_outputs = None
        if structured_output:
            structured_outputs = StructuredOutputsParams(
                json=config.get("output_schema"), disable_any_whitespace=True
            )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=config["max_tokens"],
            seed=0,
            structured_outputs=structured_outputs,
        )
        for _ in range(config["warmup_runs"]):
            llm.generate(requests[0], sampling_params=sampling, use_tqdm=False)
        torch.cuda.synchronize()
        runs = []
        for pass_index in range(config["runs"]):
            for batch_index, request in enumerate(requests):
                run = _run_request(llm, request, sampling)
                first_image = batch_index * images_per_batch + 1
                run.update(
                    {
                        "pass_index": pass_index + 1,
                        "batch_index": batch_index + 1,
                        "image_count": len(request["multi_modal_data"]["image"]),
                        "image_indices": list(
                            range(
                                first_image,
                                first_image + len(request["multi_modal_data"]["image"]),
                            )
                        ),
                    }
                )
                runs.append(run)
        warnings = sorted({run["warning"] for run in runs if run.get("warning")})
        return {
            "model": model,
            "status": "ok",
            "runtime": "vllm",
            "versions": {
                "python": platform.python_version(),
                "vllm": package_version("vllm"),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": package_version("transformers"),
            },
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(0))
            ),
            "model_load_seconds": load_seconds,
            "engine": engine_kwargs,
            "sampling": {
                "temperature": 0.0,
                "max_tokens": config["max_tokens"],
                "seed": 0,
                "structured_outputs": (
                    {
                        "json": config.get("output_schema"),
                        "disable_any_whitespace": True,
                    }
                    if structured_output
                    else None
                ),
            },
            "warnings": warnings,
            "runs": runs,
            "summary": summarize_runs(runs),
        }
    finally:
        for image in images:
            image.close()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            del llm
        gc.collect()
        torch.cuda.empty_cache()


def run_worker(model: str, config_path: Path) -> int:
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


def run_benchmark(args: argparse.Namespace) -> int:
    """Prepare inputs and coordinate the per-model worker loop."""
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = workspace_root() / "data" / "benchmarks" / "vllm" / timestamp
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

    config = build_run_config(args, images, output_dir)
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
                    f"{Path(__file__).resolve().parent.name}.benchmark",
                    "--worker-model",
                    model,
                    "--config",
                    str(config_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=Path(__file__).resolve().parent.parent,
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
        "runtime": "vllm",
        "device": device_metadata(),
        "config": config,
        "images": images,
        "models": results,
    }
    write_reports(output_dir, payload)
    print(f"\nResults written to {output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(result["status"] == "ok" for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    """Dispatch coordinator and worker invocations."""
    args = parse_args(argv)
    if args.worker_model:
        return run_worker(args.worker_model, Path(args.config))
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
