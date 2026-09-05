"""Configuration and command-line parsing for the vLLM benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "google/gemma-4-E2B-it",
)
MODEL_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3.5": {
        "prefixes": ("qwen/qwen3.5-",),
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
COCO_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017"
COCO_IMAGE_NAMES = (
    "000000000139.jpg", "000000000285.jpg", "000000000632.jpg", "000000000724.jpg",
    "000000000776.jpg", "000000000785.jpg", "000000000802.jpg", "000000000872.jpg",
    "000000000885.jpg", "000000001000.jpg", "000000001268.jpg", "000000001296.jpg",
    "000000001353.jpg", "000000001425.jpg", "000000001490.jpg", "000000001503.jpg",
    "000000001532.jpg", "000000001584.jpg", "000000001675.jpg", "000000001761.jpg",
    "000000001818.jpg", "000000001993.jpg", "000000002006.jpg", "000000002149.jpg",
    "000000002153.jpg", "000000002157.jpg", "000000002261.jpg", "000000002299.jpg",
    "000000002431.jpg", "000000002473.jpg", "000000002532.jpg", "000000002587.jpg",
    "000000002592.jpg", "000000002685.jpg", "000000002923.jpg", "000000003156.jpg",
    "000000003255.jpg", "000000003501.jpg", "000000003553.jpg", "000000003661.jpg",
)
DEFAULT_IMAGES = tuple(f"{COCO_BASE_URL}/{name}" for name in COCO_IMAGE_NAMES)
DATASET_SIZE = len(COCO_IMAGE_NAMES)
MAX_BATCHES = 10
DEFAULT_BATCHES = 10
DEFAULT_IMAGES_PER_BATCH = 4
IMAGE_SIZE = (336, 224)  # width, height
DEFAULT_KV_CACHE_MEMORY_BYTES = 4 * 1024**3


def build_prompt(images_per_batch: int) -> str:
    noun = "image" if images_per_batch == 1 else "images"
    verb = "is" if images_per_batch == 1 else "are"
    return f"""The {images_per_batch} {noun} {verb} numbered 1 through {images_per_batch} in presentation order.
For each image, enumerate the clearly visible objects and estimate each object's center point as normalized [x, y] coordinates, where [0, 0] is the top-left and [1, 1] is the bottom-right."""


def build_output_schema(images_per_batch: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "minItems": images_per_batch,
                "maxItems": images_per_batch,
                "items": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": images_per_batch,
                        },
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "center": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 1,
                                        },
                                    },
                                },
                                "required": ["name", "center"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["image", "objects"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["images"],
        "additionalProperties": False,
    }


def validate_batch_settings(
    batches: int, images_per_batch: int, dataset_size: int = DATASET_SIZE
) -> int:
    if not 1 <= batches <= MAX_BATCHES:
        raise ValueError(f"batches must be between 1 and {MAX_BATCHES}, got {batches}")
    if images_per_batch < 1:
        raise ValueError(
            f"images per batch must be greater than zero, got {images_per_batch}"
        )
    total_images = batches * images_per_batch
    if total_images > dataset_size:
        raise ValueError(
            f"batches * images per batch must not exceed the {dataset_size}-image "
            f"dataset, got {batches} * {images_per_batch} = {total_images}"
        )
    return total_images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM benchmark using 40 COCO images on Jetson Orin and Thor."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--images", nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument(
        "--batches",
        type=int,
        default=DEFAULT_BATCHES,
        help=f"Number of image batches to run, from 1 to {MAX_BATCHES}",
    )
    parser.add_argument(
        "--images-per-batch",
        type=int,
        default=DEFAULT_IMAGES_PER_BATCH,
        help="Number of images in each batch; batches × images must be at most 40",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of full configured-workload passes (default: 1)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2560)
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
    parser.add_argument(
        "--no-reclaim-memory",
        action="store_false",
        dest="reclaim_memory",
        help="Do not drop retained CUDA/page caches between models on L4T below R39",
    )
    parser.set_defaults(reclaim_memory=True)
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_model:
        if not args.config:
            parser.error("--config is required in worker mode")
        return args
    if len(args.images) != DATASET_SIZE:
        parser.error(f"--images requires exactly {DATASET_SIZE} paths or URLs")
    try:
        validate_batch_settings(args.batches, args.images_per_batch, len(args.images))
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup_runs < 0 or args.runs < 1:
        parser.error("--warmup-runs must be >= 0 and --runs must be >= 1")
    if args.max_tokens < 1 or args.max_model_len < 1:
        parser.error("token limits must be positive")
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
        "output_dir": str(output_dir),
        "models": args.models,
        "images": images,
        "dataset_size": len(args.images),
        "batches": args.batches,
        "images_per_batch": args.images_per_batch,
        "prompt": build_prompt(args.images_per_batch),
        "structured_output": args.structured_output,
        "output_schema": (
            build_output_schema(args.images_per_batch)
            if args.structured_output
            else None
        ),
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "reclaim_memory": args.reclaim_memory,
        "batch_size": args.images_per_batch,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
    }
