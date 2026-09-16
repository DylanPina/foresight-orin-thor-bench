"""vLLM-specific configuration and command-line parsing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..common.config import (
    add_workload_arguments,
    build_workload_config,
    validate_workload_arguments,
)

RUNTIME_NAME = "vllm"
MODEL_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3.5": {
        "prefixes": ("qwen/qwen3.5-",),
        "engine_kwargs": {},
    },
    "qwen3vl": {
        "prefixes": (
            "qwen/qwen3-vl-",
            "ut-amrl/foresight-qwen3vl-",
        ),
        "engine_kwargs": {},
    },
    "gemma4": {
        "prefixes": ("google/gemma-4",),
        "engine_kwargs": {
            "mm_processor_kwargs": {"max_soft_tokens": 280},

            # ---
            # Gemma 4 uses two attention head dimensions:
            #   - Sliding-attention layers: 256
            #   - Full-attention layers: 523
            # 
            # When the backend is auto, vLLM selects FlashAttention. 
            # It attempts FA4, Thor build reports that FA4 cannot handle Gemma’s required 512-dimensional heads. 
            # It then falls back to FA2, which supports at most 256, causing the following runtime error:
            # `FlashAttention forward only supports head dimension at most 256`
            # ---
            "attention_config": {"backend": "TRITON_ATTN"}, # FlashAttention forward only supports head dimension at most 256 (https://github.com/vllm-project/vllm/issues/40677)
        },
    },
}
DEFAULT_KV_CACHE_MEMORY_BYTES = 4 * 1024**3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM benchmark using 40 COCO images on Jetson Orin and Thor."
    )
    add_workload_arguments(parser)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Enable JSON Schema constrained decoding (default: disabled)",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=DEFAULT_KV_CACHE_MEMORY_BYTES,
        help="Manually size the KV cache and bypass vLLM memory profiling",
    )
    args = parser.parse_args(argv)
    if validate_workload_arguments(parser, args):
        return args
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes < 1:
        parser.error("--kv-cache-memory-bytes must be positive")
    return args


def build_run_config(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the serializable configuration shared with worker processes."""
    return {
        **build_workload_config(args, images, output_dir, args.structured_output),
        "runtime": RUNTIME_NAME,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
    }
