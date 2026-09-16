"""TensorRT Edge-LLM model execution for the shared Jetson VLM benchmark.

Run inside the ``edgellm-thor`` container (docker/Dockerfile.thor_edgellm). Each
model is exported to ONNX, built into TensorRT engines, and then measured with
one ``llm_inference`` invocation per image batch 
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from ..common.config import DATASET_SIZE, validate_batch_settings
from ..common.runner import run_benchmark, run_worker
from ..common.utils import (
    ResourceMonitor,
    command_output,
    package_version,
    slug,
    summarize_runs,
    workspace_root,
)
from .config import (
    DEFAULT_WORKSPACE_SUBDIR,
    MODEL_FAMILY_PROFILES,
    RUNTIME_NAME,
    build_run_config,
    engine_config_for,
    model_family,
    parse_args,
)
from .utils import (
    TTFT_WARNING,
    EdgeLLMTools,
    build_engines,
    export_model,
    is_measurement_start,
    load_json,
    model_workspace_lock,
    parse_profile,
    run_logged,
    write_request_file,
)

REPORT_TITLE = "TensorRT Edge-LLM Jetson benchmark"


def _edgellm_version(root: Path) -> str | None:
    version_file = root / "tensorrt_edgellm" / "_version.py"
    try:
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return command_output(["git", "-C", str(root), "describe", "--tags", "--always"])


def _tensorrt_version() -> str | None:
    version = package_version("tensorrt")
    if version:
        return version
    output = command_output(["dpkg-query", "-W", "-f=${Version}", "libnvinfer10"])
    return output or None


def _cuda_version() -> str | None:
    output = command_output(["nvcc", "--version"]) or ""
    match = re.search(r"release (\d+\.\d+)", output)
    return match.group(1) if match else None


def _run_request(
    tools: EdgeLLMTools,
    engine_dir: Path,
    request_path: Path,
    output_path: Path,
    profile_path: Path,
    warmup_runs: int,
    max_tokens: int,
    log_stream: Any,
) -> dict[str, Any]:
    """Run one llm_inference invocation and map its profile to a run record."""
    monitor = ResourceMonitor()
    measurement_start: list[float] = []

    def on_line(line: str) -> None:
        if not measurement_start and is_measurement_start(line):
            measurement_start.append(time.monotonic())

    command = [
        str(tools.llm_inference),
        "--engineDir", str(engine_dir / "llm"),
        "--multimodalEngineDir", str(engine_dir),
        "--inputFile", str(request_path),
        "--outputFile", str(output_path),
        # --dumpProfile enables stage profiling; without it the profile JSON only
        # carries wall-clock timing.
        "--dumpProfile",
        "--profileOutputFile", str(profile_path),
        "--warmup", str(warmup_runs),
        # Warmup reuses the measured request; without this the runtime serves
        # the ViT embeddings from its content-addressed cache and skips the
        # vision encoder. vLLM ran with mm_processor_cache_gb=0 for the same reason.
        "--encoderCacheBudgetBytes", "0",
        "--batchSize", "1",
        "--maxGenerateLength", str(max_tokens),
    ]
    monitor.start()
    started = time.perf_counter()
    try:
        returncode = run_logged(command, log_stream, cwd=tools.root, on_line=on_line)
    finally:
        process_elapsed = time.perf_counter() - started
    if returncode != 0:
        monitor.stop(process_elapsed)
        raise RuntimeError(f"llm_inference exited {returncode} for {request_path.name}")
    profile = load_json(profile_path)
    output = load_json(output_path)
    run = parse_profile(profile, output)
    e2e_seconds = (run["e2e_latency_ms"] or 0.0) / 1000.0
    # Only samples taken after engine load and warmup count toward power/thermals.
    resources = monitor.stop(
        e2e_seconds or process_elapsed,
        since=measurement_start[0] if measurement_start else None,
    )
    if not measurement_start:
        resources["warning"] = "; ".join(
            warning
            for warning in (
                resources.get("warning"),
                "measurement start marker not found in llm_inference output; "
                "power samples include engine load",
            )
            if warning
        )
    run.update(resources)
    run["process_elapsed_ms"] = process_elapsed * 1000.0
    return run


def run_model(model: str, config: dict[str, Any]) -> dict[str, Any]:
    """Export, build, and benchmark one model inside a worker process."""
    tools = EdgeLLMTools(Path(config["edgellm_root"]))
    tools.validate(need_export=not config.get("skip_export", False))
    image_paths = [item["path"] for item in config["images"]]
    batch_count = int(config["batches"])
    images_per_batch = int(config["images_per_batch"])
    expected_image_count = validate_batch_settings(
        batch_count, images_per_batch, int(config.get("dataset_size", DATASET_SIZE))
    )
    if len(image_paths) != expected_image_count:
        raise ValueError(
            f"configuration requires {expected_image_count} prepared images, "
            f"got {len(image_paths)}"
        )
    family = model_family(model)
    profile = MODEL_FAMILY_PROFILES.get(family or "", {})
    export_args = tuple(profile.get("export_args", ()))
    engine_config = engine_config_for(model, config["engine"], images_per_batch)

    output_dir = Path(config["output_dir"])
    model_dir = output_dir / "models" / slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(config["workspace_dir"]) / slug(model)
    onnx_dir = workspace / "onnx"
    engine_dir = workspace / "engines"
    force = bool(config.get("rebuild_engines", False))

    with model_workspace_lock(workspace):
        return _run_model_locked(
            tools, model, config, family, export_args, engine_config,
            model_dir, onnx_dir, engine_dir, force, image_paths, images_per_batch,
        )


def _run_model_locked(
    tools: EdgeLLMTools,
    model: str,
    config: dict[str, Any],
    family: str | None,
    export_args: tuple[str, ...],
    engine_config: dict[str, Any],
    model_dir: Path,
    onnx_dir: Path,
    engine_dir: Path,
    force: bool,
    image_paths: list[str],
    images_per_batch: int,
) -> dict[str, Any]:
    """Export, build, and measure one model while holding its workspace lock."""
    with (model_dir / "build.log").open("w", encoding="utf-8") as build_log:
        export_info = export_model(
            tools, model, onnx_dir, export_args, build_log, force,
            bool(config.get("skip_export", False)),
        )
        build_info = build_engines(
            tools, onnx_dir, engine_dir, engine_config, build_log, force,
        )

    requests_dir = model_dir / "requests"
    requests_dir.mkdir(exist_ok=True)
    request_paths: list[tuple[Path, list[str]]] = []
    for start in range(0, len(image_paths), images_per_batch):
        batch_paths = image_paths[start : start + images_per_batch]
        request_path = requests_dir / f"batch_{len(request_paths) + 1:02d}.json"
        write_request_file(request_path, batch_paths, config["prompt"], config["max_tokens"])
        request_paths.append((request_path, batch_paths))

    runs = []
    with (model_dir / "inference.log").open("w", encoding="utf-8") as inference_log:
        for pass_index in range(int(config["runs"])):
            for batch_index, (request_path, batch_paths) in enumerate(request_paths):
                stem = f"pass_{pass_index + 1:02d}_batch_{batch_index + 1:02d}"
                run = _run_request(
                    tools,
                    engine_dir,
                    request_path,
                    requests_dir / f"{stem}_output.json",
                    requests_dir / f"{stem}_profile.json",
                    int(config["warmup_runs"]),
                    int(config["max_tokens"]),
                    inference_log,
                )
                first_image = batch_index * images_per_batch + 1
                run.update(
                    {
                        "pass_index": pass_index + 1,
                        "batch_index": batch_index + 1,
                        "image_count": len(batch_paths),
                        "image_indices": list(
                            range(first_image, first_image + len(batch_paths))
                        ),
                    }
                )
                runs.append(run)
    warnings = sorted({run["warning"] for run in runs if run.get("warning")})
    warnings.append(TTFT_WARNING)
    return {
        "model": model,
        "model_family": family,
        "status": "ok",
        "runtime": RUNTIME_NAME,
        "versions": {
            "python": platform.python_version(),
            "tensorrt_edgellm": _edgellm_version(tools.root),
            "tensorrt": _tensorrt_version(),
            "cuda": _cuda_version(),
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        },
        "cuda_device": command_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ),
        "precision": "fp16",
        "export": {"args": list(export_args), **export_info},
        "engine_build": build_info,
        "export_seconds": export_info["seconds"],
        "engine_build_seconds": build_info["seconds"],
        "engine": {
            **engine_config,
            "engine_dir": str(engine_dir),
            "onnx_dir": str(onnx_dir),
        },
        "runtime_args": {
            "encoder_cache_budget_bytes": 0,
            "enable_context_reuse": False,
            "warmup": int(config["warmup_runs"]),
        },
        "sampling": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": config["max_tokens"],
            "enable_thinking": False,
            "structured_outputs": None,
        },
        "warnings": warnings,
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def _build_config(args: argparse.Namespace, images: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    root = workspace_root()
    workspace_dir = (
        args.workspace_dir if args.workspace_dir is not None else root / DEFAULT_WORKSPACE_SUBDIR
    ).expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return build_run_config(args, images, output_dir, workspace_dir)


def main(argv: list[str] | None = None) -> int:
    """Dispatch coordinator and worker invocations."""
    args = parse_args(argv)
    if args.worker_model:
        return run_worker(args.worker_model, Path(args.config), run_model)
    try:
        EdgeLLMTools(args.edgellm_root).validate(need_export=not args.skip_export)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    return run_benchmark(
        args,
        worker_module="benchmark.edgellm.benchmark",
        runtime=RUNTIME_NAME,
        runtime_dir="edgellm",
        report_title=REPORT_TITLE,
        build_config=_build_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
