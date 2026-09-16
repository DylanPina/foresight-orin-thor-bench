"""TensorRT Edge-LLM export, engine build, request, and profile helpers."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator

from ..common.utils import load_json, run_logged, write_json

VISUAL_STAGE_IDS = ("multimodal_processing", "vision_encoder", "audio_encoder")
PREFILL_STAGE_ID = "llm_prefill"
GENERATION_STAGE_ID = "llm_generation"
TTFT_WARNING = (
    "Edge-LLM TTFT is approximated from GPU stage timings (vision encoder + "
    "prefill), not measured at the first emitted token"
)


class EdgeLLMTools:
    """Paths to the Edge-LLM binaries and export CLI inside the container."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        examples = self.root / "build" / "examples"
        self.llm_build = examples / "llm" / "llm_build"
        self.llm_inference = examples / "llm" / "llm_inference"
        self.visual_build = examples / "multimodal" / "visual_build"
        venv = os.environ.get("EDGELLM_VENV")
        candidate = Path(venv) / "bin" / "tensorrt-edgellm-export" if venv else None
        found = shutil.which("tensorrt-edgellm-export")
        self.export = (
            candidate if candidate and candidate.is_file() else Path(found) if found else None
        )

    def validate(self, need_export: bool) -> None:
        missing = [
            str(path)
            for path in (self.llm_build, self.llm_inference, self.visual_build)
            if not (path.is_file() and os.access(path, os.X_OK))
        ]
        if need_export and self.export is None:
            missing.append("tensorrt-edgellm-export (not on PATH)")
        if missing:
            raise RuntimeError(
                "TensorRT Edge-LLM tools are missing; run inside the edgellm-thor "
                "container built from docker/Dockerfile.thor_edgellm. Missing: "
                + ", ".join(missing)
            )


def _has_files(directory: Path, pattern: str) -> bool:
    return directory.is_dir() and any(directory.glob(pattern))


def onnx_ready(onnx_dir: Path) -> bool:
    return _has_files(onnx_dir / "llm", "*.onnx") and _has_files(
        onnx_dir / "visual", "*.onnx"
    )


BUILD_PARAMS_FILE = "foresight_build_params.json"
LOCK_FILE = ".foresight.lock"


@contextmanager
def model_workspace_lock(workspace: Path) -> Iterator[None]:
    """Serialize benchmark processes that share one model's ONNX/engine cache.

    Two runs pointed at the same workspace (for example from two containers)
    would otherwise rebuild engines underneath each other's llm_inference.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / LOCK_FILE
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"Waiting for another benchmark process holding {lock_path} ...",
                flush=True,
            )
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def engines_ready(engine_dir: Path, engine: dict[str, Any] | None = None) -> bool:
    """True when both engines exist and, if given, were built with ``engine``."""
    # llm_build writes <engine_dir>/llm/, visual_build writes <engine_dir>/visual/.
    if not (
        _has_files(engine_dir / "llm", "*.engine")
        and _has_files(engine_dir / "visual", "*.engine")
    ):
        return False
    if engine is None:
        return True
    try:
        return load_json(engine_dir / BUILD_PARAMS_FILE) == engine
    except (OSError, ValueError):
        return False


def export_model(
    tools: EdgeLLMTools,
    model: str,
    onnx_dir: Path,
    export_args: tuple[str, ...],
    log_stream: IO[str] | None,
    force: bool,
    skip_export: bool,
) -> dict[str, Any]:
    """Export a Hugging Face checkpoint to onnx/llm and onnx/visual (CPU)."""
    if onnx_ready(onnx_dir) and not force:
        return {"performed": False, "seconds": 0.0, "onnx_dir": str(onnx_dir)}
    if skip_export:
        raise RuntimeError(f"--skip-export set but ONNX export is missing at {onnx_dir}")
    assert tools.export is not None
    staging_dir = onnx_dir.with_name(f"{onnx_dir.name}.exporting-{os.getpid()}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode = run_logged(
        [str(tools.export), model, str(staging_dir), *export_args],
        log_stream,
        cwd=tools.root,
    )
    seconds = time.perf_counter() - started
    if returncode != 0 or not onnx_ready(staging_dir):
        raise RuntimeError(
            f"tensorrt-edgellm-export exited {returncode} without producing "
            f"onnx/llm and onnx/visual under {staging_dir}"
        )
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    staging_dir.rename(onnx_dir)
    return {"performed": True, "seconds": seconds, "onnx_dir": str(onnx_dir)}


def build_engines(
    tools: EdgeLLMTools,
    onnx_dir: Path,
    engine_dir: Path,
    engine: dict[str, Any],
    log_stream: IO[str] | None,
    force: bool,
) -> dict[str, Any]:
    """Build the LLM and visual TensorRT engines for one exported model."""
    if engines_ready(engine_dir, engine) and not force:
        return {"performed": False, "seconds": 0.0, "engine_dir": str(engine_dir)}
    # Build into a staging directory and rename it into place so a reader never
    # sees a partially written engine or safetensors file.
    staging_dir = engine_dir.with_name(f"{engine_dir.name}.building-{os.getpid()}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    llm_command = [
        str(tools.llm_build),
        "--onnxDir", str(onnx_dir / "llm"),
        "--engineDir", str(staging_dir / "llm"),
        "--maxBatchSize", str(engine["max_batch_size"]),
        "--maxInputLen", str(engine["max_input_len"]),
        "--maxKVCacheCapacity", str(engine["max_kv_cache_capacity"]),
    ]
    returncode = run_logged(llm_command, log_stream, cwd=tools.root)
    if returncode != 0:
        raise RuntimeError(f"llm_build exited {returncode}")
    visual_command = [
        str(tools.visual_build),
        "--onnxDir", str(onnx_dir / "visual"),
        "--engineDir", str(staging_dir),
        "--minImageTokens", str(engine["min_image_tokens"]),
        "--maxImageTokens", str(engine["max_image_tokens"]),
        "--maxImageTokensPerImage", str(engine["max_image_tokens_per_image"]),
    ]
    returncode = run_logged(visual_command, log_stream, cwd=tools.root)
    if returncode != 0:
        raise RuntimeError(f"visual_build exited {returncode}")
    if not engines_ready(staging_dir):
        raise RuntimeError(f"engine build finished but no engines found under {staging_dir}")
    write_json(staging_dir / BUILD_PARAMS_FILE, engine)
    if engine_dir.exists():
        shutil.rmtree(engine_dir)
    staging_dir.rename(engine_dir)
    return {"performed": True, "seconds": time.perf_counter() - started, "engine_dir": str(engine_dir)}


def write_request_file(
    path: Path, image_paths: list[str], prompt: str, max_tokens: int
) -> None:
    """Write one greedy, thinking-off, multi-image request for llm_inference."""
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(Path(image).resolve())} for image in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    write_json(
        path,
        {
            "batch_size": 1,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_generate_length": max_tokens,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "requests": [{"messages": [{"role": "user", "content": content}]}],
        },
    )


_STAGE_START = re.compile(r"Processing \d+ batched requests")


def is_measurement_start(line: str) -> bool:
    """True for the llm_inference log line that opens the wall-clock window."""
    return _STAGE_START.search(line) is not None


def _stage_gpu_ms(profile: dict[str, Any], stage_ids: tuple[str, ...]) -> float | None:
    total = 0.0
    seen = False
    for stage in profile.get("stages", []):
        if stage.get("stage_id") in stage_ids:
            total += float(stage.get("total_gpu_time_ms", 0.0))
            seen = True
    return total if seen else None


def parse_profile(profile: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    """Map an llm_inference profile/output pair onto the vLLM per-run schema."""
    wall_clock = profile.get("wall_clock", {})
    prefill = profile.get("prefill", {})
    generation = profile.get("generation", {})
    multimodal = profile.get("multimodal", {})
    # Single-rank runs write the memory summary at the top level; TP runs nest it.
    memory = profile.get("memory") or profile
    responses = output.get("responses", [])
    response = responses[0] if responses else {}

    e2e_ms = wall_clock.get("total_time_ms")
    output_tokens = int(wall_clock.get("generated_tokens") or generation.get("generated_tokens") or 0)
    visual_ms = _stage_gpu_ms(profile, VISUAL_STAGE_IDS)
    prefill_ms = prefill.get("average_time_per_run_ms")
    ttft_ms = (
        (visual_ms or 0.0) + float(prefill_ms) if prefill_ms is not None else None
    )
    decode_tps = generation.get("tokens_per_second")
    tpot_ms = generation.get("average_time_per_token_ms")
    return {
        "ttft_ms": ttft_ms,
        "e2e_latency_ms": float(e2e_ms) if e2e_ms is not None else None,
        "output_tokens": output_tokens,
        "output_throughput_tokens_s": wall_clock.get("tokens_per_second"),
        "decode_throughput_tokens_s": float(decode_tps) if decode_tps else None,
        "tpot_ms": float(tpot_ms) if tpot_ms else None,
        "finish_reason": response.get("finish_reason"),
        "generated_text": response.get("output_text"),
        "torch_peak_allocated_mb": None,
        "torch_peak_reserved_mb": None,
        # Thor's integrated GPU reports gpu_memory_metric "unavailable" (0 MB).
        "runtime_peak_gpu_memory_mb": (
            memory.get("peak_gpu_memory_mb")
            if memory.get("gpu_memory_metric") != "unavailable"
            else None
        ),
        "runtime_peak_cpu_memory_mb": memory.get("peak_cpu_memory_mb"),
        "vision_encoder_ms": visual_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens": prefill.get("computed_tokens"),
        "image_tokens": multimodal.get("total_image_tokens"),
        "profile": profile,
    }
