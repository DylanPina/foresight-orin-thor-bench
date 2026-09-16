"""TensorRT Edge-LLM-specific configuration and command-line parsing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ..common.config import (
    add_workload_arguments,
    build_workload_config,
    validate_workload_arguments,
)
from ..common.config import model_family as _model_family

RUNTIME_NAME = "tensorrt-edgellm"
DEFAULT_EDGELLM_ROOT = Path(os.environ.get("EDGELLM_ROOT", "/opt/TensorRT-Edge-LLM"))
DEFAULT_WORKSPACE_SUBDIR = Path("data") / "edgellm"

# Engine profile limits. ``max_input_len`` covers the chat template, the prompt,
# and up to ``images_per_batch`` images of image tokens; the KV capacity mirrors
# vLLM's ``--max-model-len`` so both runtimes reserve the same sequence budget.
DEFAULT_MAX_INPUT_LEN = 2048
DEFAULT_MIN_IMAGE_TOKENS = 8
DEFAULT_MAX_IMAGE_TOKENS_PER_IMAGE = 512

MODEL_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3.5": {
        "prefixes": ("qwen/qwen3.5-",),
        "export_args": (),
    },
    "qwen3vl": {
        "prefixes": (
            "qwen/qwen3-vl-",
            "ut-amrl/foresight-qwen3vl-",
        ),
        "export_args": (),
    },
    "gemma4": {
        "prefixes": ("google/gemma-4",),
        # Gemma 4 E2B/E4B ship an audio encoder; the benchmark is image-only.
        "export_args": ("--skip-audio",),
        # The Gemma 4 runner caps soft tokens per image at the visual engine's
        # --maxImageTokensPerImage rather than the processor's max_soft_tokens,
        # so pin it to the 280 the vLLM benchmark used (mm_processor_kwargs).
        "max_image_tokens_per_image": 280,
    },
}


def engine_config_for(model: str, engine: dict[str, Any], images_per_batch: int) -> dict[str, Any]:
    """Apply per-family overrides to the shared engine build parameters."""
    profile = MODEL_FAMILY_PROFILES.get(model_family(model) or "", {})
    result = dict(engine)
    per_image = profile.get("max_image_tokens_per_image")
    if per_image is not None and per_image < result["max_image_tokens_per_image"]:
        result["max_image_tokens_per_image"] = per_image
        result["max_image_tokens"] = images_per_batch * per_image
    return result


def model_family(model: str) -> str | None:
    return _model_family(model, MODEL_FAMILY_PROFILES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline TensorRT Edge-LLM benchmark using 40 COCO images on Jetson "
            "Thor. Run inside the edgellm-thor container."
        )
    )
    add_workload_arguments(parser)
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=DEFAULT_MAX_INPUT_LEN,
        help="Maximum prefill length built into the LLM engine",
    )
    parser.add_argument(
        "--max-image-tokens-per-image",
        type=int,
        default=DEFAULT_MAX_IMAGE_TOKENS_PER_IMAGE,
        help="Visual engine per-image token limit",
    )
    parser.add_argument(
        "--edgellm-root",
        type=Path,
        default=DEFAULT_EDGELLM_ROOT,
        help="TensorRT-Edge-LLM checkout with a completed build/ directory",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        help="Where ONNX exports and TensorRT engines are cached "
        f"(default: <workspace>/{DEFAULT_WORKSPACE_SUBDIR})",
    )
    parser.add_argument(
        "--rebuild-engines",
        action="store_true",
        help="Re-export ONNX and rebuild engines even when cached copies exist",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Fail instead of exporting when the ONNX directory is missing",
    )
    args = parser.parse_args(argv)
    if validate_workload_arguments(parser, args):
        return args
    if args.max_input_len < 1:
        parser.error("--max-input-len must be positive")
    if args.max_input_len + args.max_tokens > args.max_model_len:
        parser.error(
            "--max-input-len + --max-tokens must not exceed --max-model-len "
            f"({args.max_input_len} + {args.max_tokens} > {args.max_model_len})"
        )
    if args.max_image_tokens_per_image < 1:
        parser.error("--max-image-tokens-per-image must be positive")
    return args


def build_run_config(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    output_dir: Path,
    workspace_dir: Path,
) -> dict[str, Any]:
    """Build the serializable configuration shared with worker processes."""
    return {
        # Edge-LLM has no JSON Schema constrained decoding; the workload keeps
        # structured_output=False for schema parity with the vLLM run.
        **build_workload_config(args, images, output_dir),
        "runtime": RUNTIME_NAME,
        "edgellm_root": str(args.edgellm_root.expanduser().resolve()),
        "workspace_dir": str(workspace_dir),
        "rebuild_engines": args.rebuild_engines,
        "skip_export": args.skip_export,
        "engine": {
            "max_batch_size": 1,
            "max_input_len": args.max_input_len,
            "max_kv_cache_capacity": args.max_model_len,
            "min_image_tokens": DEFAULT_MIN_IMAGE_TOKENS,
            "max_image_tokens": args.images_per_batch * args.max_image_tokens_per_image,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
