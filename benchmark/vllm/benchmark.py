"""vLLM model execution for the shared Jetson VLM benchmark."""

from __future__ import annotations

import gc
import platform
import time
from pathlib import Path
from typing import Any

from ..common.config import DATASET_SIZE, model_family, validate_batch_settings
from ..common.runner import run_benchmark, run_worker
from ..common.utils import ResourceMonitor, package_version, summarize_runs
from .config import MODEL_FAMILY_PROFILES, RUNTIME_NAME, build_run_config, parse_args

REPORT_TITLE = "vLLM Jetson benchmark"


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
    family = model_family(model, MODEL_FAMILY_PROFILES)
    if family is not None:
        engine_kwargs.update(MODEL_FAMILY_PROFILES[family].get("engine_kwargs", {}))

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
            "model_family": family,
            "status": "ok",
            "runtime": RUNTIME_NAME,
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


def main(argv: list[str] | None = None) -> int:
    """Dispatch coordinator and worker invocations."""
    args = parse_args(argv)
    if args.worker_model:
        return run_worker(args.worker_model, Path(args.config), run_model)
    return run_benchmark(
        args,
        worker_module="benchmark.vllm.benchmark",
        runtime=RUNTIME_NAME,
        runtime_dir="vllm",
        report_title=REPORT_TITLE,
        build_config=build_run_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
