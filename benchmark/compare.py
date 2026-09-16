"""Compare a vLLM benchmark run against a TensorRT Edge-LLM benchmark run.

Both runs must come from ``benchmark.vllm`` and ``benchmark.edgellm``, which
share the workload and the ``results.json`` schema.

    python3 -m benchmark.compare \
        --vllm data/benchmarks/vllm/<timestamp> \
        --edgellm data/benchmarks/edgellm/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common.runner import default_output_dir
from .common.utils import load_json, percentile

WORKLOAD_KEYS = ("batches", "images_per_batch", "max_tokens", "max_model_len", "prompt")

# (label, summary key, unit, lower_is_better)
METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("Output tokens p50", "output_tokens_p50", "tok", None),
    ("Output tokens total", "output_tokens_total", "tok", None),
    ("Batches at max tokens", "batches_at_max_tokens", "n", None),
    ("TTFT p50", "ttft_ms_p50", "ms", True),
    ("E2E p50", "e2e_latency_ms_p50", "ms", True),
    ("E2E p95", "e2e_latency_ms_p95", "ms", True),
    ("Time/image", "effective_time_per_image_ms", "ms", True),
    ("Output tok/s", "aggregate_output_throughput_tokens_s", "tok/s", False),
    ("Decode tok/s", "decode_throughput_tokens_s_p50", "tok/s", False),
    ("TPOT p50", "tpot_ms_p50", "ms", True),
    ("Peak RAM", "system_ram_used_mb_peak", "MB", True),
    ("GPU W avg", "gpu_power_w_average", "W", True),
    ("Board W avg", "board_power_w_average", "W", True),
    ("Board J/image", "board_energy_j_per_image", "J", True),
    ("Max temp", "temperature_c_max", "°C", True),
)


def load_run(directory: Path) -> dict[str, Any]:
    path = directory / "results.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return load_json(path)


def _fmt(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):,.{digits}f}"


def _ratio(vllm: Any, edge: Any, lower_is_better: bool | None) -> float | None:
    if vllm is None or edge is None or lower_is_better is None:
        return None
    vllm, edge = float(vllm), float(edge)
    if lower_is_better:
        return vllm / edge if edge else None
    return edge / vllm if vllm else None


def summary_with_derived(result: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
    summary = dict(result.get("summary", {}))
    images = summary.get("measured_images")
    energy = summary.get("board_energy_j_estimate")
    summary["board_energy_j_per_image"] = (
        energy / images if energy is not None and images else None
    )
    tokens = sorted(int(run["output_tokens"]) for run in result.get("runs", []))
    summary["output_tokens_p50"] = percentile(tokens, 0.50) if tokens else None
    summary["output_tokens_total"] = sum(tokens) if tokens else None
    summary["batches_at_max_tokens"] = (
        sum(1 for t in tokens if max_tokens is not None and t >= max_tokens) if tokens else None
    )
    return summary


def sample_text(result: dict[str, Any]) -> str:
    for run in result.get("runs", []):
        text = run.get("generated_text")
        if text:
            return " ".join(text.split())[:240]
    return ""


def header_lines(label: str, payload: dict[str, Any]) -> list[str]:
    device = payload.get("device", {})
    versions: dict[str, Any] = {}
    for result in payload.get("models", []):
        versions = result.get("versions") or {}
        if versions:
            break
    version_text = ", ".join(
        f"{key} {value}"
        for key, value in versions.items()
        if key in ("vllm", "tensorrt_edgellm", "tensorrt", "torch", "cuda", "transformers")
        and value
    )
    return [
        f"- **{label}**: created {payload.get('created_at')}, "
        f"device `{device.get('product_model') or 'unknown'}`, L4T {device.get('l4t_release') or '?'}, "
        f"power mode `{(device.get('power_mode') or 'unknown').replace(chr(10), ' | ')}`, "
        f"container `{device.get('container_base')}`"
        + (f", versions: {version_text}" if version_text else ""),
    ]


def compare(vllm_payload: dict[str, Any], edge_payload: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    vllm_cfg, edge_cfg = vllm_payload["config"], edge_payload["config"]
    mismatches = [
        f"{key}: vLLM={vllm_cfg.get(key)!r} Edge-LLM={edge_cfg.get(key)!r}"
        for key in WORKLOAD_KEYS
        if vllm_cfg.get(key) != edge_cfg.get(key)
    ]
    vllm_models = {result["model"]: result for result in vllm_payload["models"]}
    edge_models = {result["model"]: result for result in edge_payload["models"]}
    models = [m for m in vllm_models if m in edge_models]
    only_vllm = sorted(set(vllm_models) - set(edge_models))
    only_edge = sorted(set(edge_models) - set(vllm_models))

    lines = [
        "# vLLM vs TensorRT Edge-LLM on Jetson",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Workload: {vllm_cfg['batches']} batches × {vllm_cfg['images_per_batch']} images, "
        f"max {vllm_cfg['max_tokens']} tokens, max model len {vllm_cfg['max_model_len']}",
        *header_lines("vLLM", vllm_payload),
        *header_lines("Edge-LLM", edge_payload),
    ]
    if mismatches:
        lines += ["", "**Warning: workload settings differ**", *[f"- {m}" for m in mismatches]]
    if only_vllm or only_edge:
        lines += ["", f"- Models only in vLLM run: {', '.join(only_vllm) or 'none'}",
                  f"- Models only in Edge-LLM run: {', '.join(only_edge) or 'none'}"]
    max_tokens = int(vllm_cfg["max_tokens"])
    lines += [
        "",
        "Speedup is vLLM ÷ Edge-LLM for latency/energy (>1 means Edge-LLM is faster/leaner) "
        "and Edge-LLM ÷ vLLM for throughput (>1 means Edge-LLM is faster). "
        "See the notes below the table before quoting any row.",
        "",
        "| Model | Metric | vLLM | Edge-LLM | Speedup |",
        "|---|---|---:|---:|---:|",
    ]
    rows: list[dict[str, Any]] = []
    length_notes: list[str] = []
    for model in models:
        vllm_result, edge_result = vllm_models[model], edge_models[model]
        if vllm_result["status"] != "ok" or edge_result["status"] != "ok":
            lines.append(
                f"| {model} | status | {vllm_result['status']} | {edge_result['status']} | |"
            )
            rows.append({"model": model, "metric": "status",
                         "vllm": vllm_result["status"], "edgellm": edge_result["status"], "speedup": ""})
            continue
        vllm_summary = summary_with_derived(vllm_result, max_tokens)
        edge_summary = summary_with_derived(edge_result, max_tokens)
        for label, key, unit, lower in METRICS:
            v, e = vllm_summary.get(key), edge_summary.get(key)
            ratio = _ratio(v, e, lower)
            digits = 0 if unit in ("tok", "n") else 2
            lines.append(
                f"| {model} | {label} ({unit}) | {_fmt(v, digits)} | {_fmt(e, digits)} | "
                f"{_fmt(ratio) + '×' if ratio is not None else ''} |"
            )
            rows.append({"model": model, "metric": key, "unit": unit,
                         "vllm": v, "edgellm": e, "speedup": ratio})
        v_tot, e_tot = vllm_summary["output_tokens_total"], edge_summary["output_tokens_total"]
        if v_tot and e_tot and abs(v_tot - e_tot) / max(v_tot, e_tot) > 0.02:
            length_notes.append(
                f"`{model}`: vLLM generated {v_tot:,} output tokens over the run, Edge-LLM "
                f"{e_tot:,} ({v_tot / e_tot:.2f}×). Its E2E, Time/image, and J/image rows "
                "reflect that length difference as well as runtime speed."
            )
    lines += [
        "",
        "## Notes — read before quoting",
        "",
        "- **Per-token rows are the like-for-like comparison** (Decode tok/s, TPOT). "
        "E2E latency, Time/image, and J/image scale with how many tokens each runtime "
        "generated. Both runtimes decode greedily, but FP16 vs BF16 numerics and different "
        "kernels mean a model that stops at end-of-sequence can stop at a different point "
        "on each runtime; the *Output tokens* rows show the actual lengths. Where every "
        "batch hits the max-token cap the lengths are identical and E2E is directly comparable.",
        *[f"  - {note}" for note in length_notes],
        "- **TTFT is not measured the same way.** vLLM's TTFT is wall-clock from request "
        "submission to the first token and includes CPU image preprocessing, tokenization, "
        "and scheduling. Edge-LLM's TTFT is the sum of the vision-encoder and prefill GPU "
        "stage times from its profiler and excludes CPU-side work. The TTFT speedup therefore "
        "overstates Edge-LLM's advantage; treat it as an upper bound.",
        "- **Precision differs.** vLLM ran the checkpoints in their native BF16; Edge-LLM ran "
        "FP16 engines (the only precision Edge-LLM supports for the Qwen3-VL family).",
        "- **Peak RAM is system-wide** (`RAM used` from tegrastats, which includes the OS, "
        "other containers, and the idle baseline), not the runtime's own allocation. vLLM's "
        "figure also includes its fixed 4 GiB KV-cache reservation.",
        "- **Power and energy** are tegrastats rail readings (VDD_GPU, VIN) sampled at 100 ms "
        "over the measured window only: vLLM around each `generate` call, Edge-LLM from the "
        "runtime's `Processing … batched requests` log line onward (engine load and warmup excluded).",
        "- **Each Edge-LLM batch is a fresh process** (engine load excluded from all metrics, "
        "encoder-embedding cache disabled); vLLM keeps one engine resident per model with "
        "prefix caching and the multimodal processor cache disabled.",
        "- Prompts are identical across runtimes (same 40 images, same text, same chat template "
        "settings, thinking disabled); Gemma 4 is capped at 280 soft tokens per image on both.",
    ]
    lines += ["", "## Sample outputs (first batch)", ""]
    for model in models:
        lines += [f"### {model}", "",
                  f"- vLLM: `{sample_text(vllm_models[model]) or 'n/a'}`",
                  f"- Edge-LLM: `{sample_text(edge_models[model]) or 'n/a'}`", ""]
    return lines, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vllm", type=Path, required=True, help="vLLM results directory")
    parser.add_argument("--edgellm", type=Path, required=True, help="Edge-LLM results directory")
    parser.add_argument("--output", type=Path, help="Directory for comparison.md/.csv "
                        "(default: data/benchmarks/comparison/<timestamp>)")
    args = parser.parse_args(argv)

    vllm_payload = load_run(args.vllm)
    edge_payload = load_run(args.edgellm)
    lines, rows = compare(vllm_payload, edge_payload)

    output = args.output
    if output is None:
        output = default_output_dir("comparison")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "metric", "unit", "vllm", "edgellm", "speedup"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"\nComparison written to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
