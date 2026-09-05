"""Shared IO, telemetry, statistics, and reporting helpers."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .config import DATASET_SIZE, IMAGE_SIZE


def percentile(values: Iterable[float | None], quantile: float) -> float | None:
    """Return a linearly interpolated percentile, ignoring missing values."""
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def parse_tegrastats_line(line: str) -> dict[str, float]:
    """Extract comparable instantaneous readings from a tegrastats line."""
    result: dict[str, float] = {}
    patterns = {
        "ram_used_mb": r"\bRAM\s+(\d+)/\d+MB",
        "gpu_power_mw": r"\bVDD_GPU\s+(\d+)mW/",
        "board_power_mw": r"\bVIN\s+(\d+)mW/",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            result[key] = float(match.group(1))
    temperatures = [
        float(value)
        for value in re.findall(
            r"\b(?:cpu|gpu|tj|soc\w*)@(-?\d+(?:\.\d+)?)C", line, flags=re.I
        )
    ]
    if temperatures:
        result["temperature_c"] = max(temperatures)
    return result


def proc_ram_used_mb() -> float | None:
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            fields[name] = int(value.strip().split()[0])
        return (fields["MemTotal"] - fields["MemAvailable"]) / 1024.0
    except (OSError, KeyError, ValueError):
        return None


class ResourceMonitor:
    """Collect system RAM, power, and temperature while one request runs."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float]] = []
        self.warning: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        tegrastats = shutil.which("tegrastats")
        if tegrastats:
            self._thread = threading.Thread(
                target=self._read_tegrastats, args=(tegrastats,), daemon=True
            )
        else:
            self.warning = "tegrastats is unavailable; power and thermal metrics are null"
            self._thread = threading.Thread(target=self._read_proc, daemon=True)
        self._thread.start()

    def _read_tegrastats(self, executable: str) -> None:
        try:
            self._process = subprocess.Popen(
                [executable, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                sample = parse_tegrastats_line(line)
                sample["sample_time"] = time.monotonic()
                proc_ram = proc_ram_used_mb()
                if proc_ram is not None:
                    sample["proc_ram_used_mb"] = proc_ram
                self.samples.append(sample)
                if self._stop.is_set():
                    break
        except OSError as exc:
            self.warning = f"tegrastats could not be started: {exc}"
            self._read_proc()

    def _read_proc(self) -> None:
        while not self._stop.is_set():
            sample = {"sample_time": time.monotonic()}
            ram = proc_ram_used_mb()
            if ram is not None:
                sample["proc_ram_used_mb"] = ram
            self.samples.append(sample)
            self._stop.wait(self.interval_ms / 1000.0)

    def stop(self, duration_seconds: float) -> dict[str, Any]:
        self._stop.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._process and self._process.poll() is None:
            self._process.kill()

        def values(key: str) -> list[float]:
            return [sample[key] for sample in self.samples if key in sample]

        gpu_mw = values("gpu_power_mw")
        board_mw = values("board_power_mw")
        ram = values("ram_used_mb") or values("proc_ram_used_mb")
        temperatures = values("temperature_c")
        gpu_average_w = sum(gpu_mw) / len(gpu_mw) / 1000.0 if gpu_mw else None
        board_average_w = sum(board_mw) / len(board_mw) / 1000.0 if board_mw else None
        return {
            "sample_count": len(self.samples),
            "system_ram_used_mb_peak": max(ram) if ram else None,
            "gpu_power_w_average": gpu_average_w,
            "gpu_power_w_peak": max(gpu_mw) / 1000.0 if gpu_mw else None,
            "gpu_energy_j_estimate": (
                gpu_average_w * duration_seconds if gpu_average_w is not None else None
            ),
            "board_power_w_average": board_average_w,
            "board_power_w_peak": max(board_mw) / 1000.0 if board_mw else None,
            "board_energy_j_estimate": (
                board_average_w * duration_seconds
                if board_average_w is not None
                else None
            ),
            "temperature_c_max": max(temperatures) if temperatures else None,
            "warning": self.warning,
        }


def workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "container").is_file() and (candidate / "docker").is_dir():
            return candidate
    return Path.cwd()


def _download(source: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(
            source, headers={"User-Agent": "foresight-vllm-benchmark/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
        return
    local = Path(source).expanduser().resolve()
    if not local.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    shutil.copyfile(local, destination)


def prepare_images(sources: list[str], directory: Path) -> list[dict[str, Any]]:
    from PIL import Image

    if not 1 <= len(sources) <= DATASET_SIZE:
        raise ValueError(
            f"between 1 and {DATASET_SIZE} images are required, got {len(sources)}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        original = directory / f"source_{index:02d}"
        resized = directory / f"image_{index:02d}.png"
        _download(source, original)
        try:
            with Image.open(original) as image:
                image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS).save(
                    resized, format="PNG", optimize=True
                )
        finally:
            original.unlink(missing_ok=True)
        manifest.append(
            {
                "index": index,
                "source": source,
                "path": str(resized.resolve()),
                "width": IMAGE_SIZE[0],
                "height": IMAGE_SIZE[1],
                "sha256": hashlib.sha256(resized.read_bytes()).hexdigest(),
            }
        )
    return manifest


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").replace("\0", "").strip()
    except OSError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _l4t_major() -> int | None:
    release = _read_text("/etc/nv_tegra_release")
    match = re.search(r"# R(\d+) \(release\)", release or "")
    return int(match.group(1)) if match else None


def reclaim_jetson_memory(enabled: bool) -> dict[str, Any]:
    """Release Thor RM/page-cache memory between vLLM worker processes."""
    major = _l4t_major()
    result: dict[str, Any] = {
        "enabled": enabled,
        "l4t_major": major,
        "attempted": False,
        "reclaimed": False,
    }
    if not enabled or major is None or major >= 39:
        return result
    drop_caches = Path("/proc/sys/vm/drop_caches")
    if os.geteuid() != 0 or not os.access(drop_caches, os.W_OK):
        result["warning"] = (
            "L4T below R39 retains CUDA memory after worker exit, but this process "
            "cannot write /proc/sys/vm/drop_caches. Run the benchmark in the "
            "privileged repository container or reclaim caches on the host."
        )
        return result
    result["attempted"] = True
    before = proc_ram_used_mb()
    try:
        os.sync()
        with drop_caches.open("w", encoding="utf-8") as stream:
            stream.write("3\n")
        after = proc_ram_used_mb()
        result.update(
            {
                "reclaimed": True,
                "ram_used_mb_before": before,
                "ram_used_mb_after": after,
                "ram_reclaimed_mb": (
                    before - after if before is not None and after is not None else None
                ),
            }
        )
    except OSError as exc:
        result["warning"] = f"Jetson memory reclamation failed: {exc}"
    return result


def device_metadata() -> dict[str, Any]:
    l4t_line = _read_text("/etc/nv_tegra_release")
    l4t_match = re.search(r"# R(\d+) \(release\), REVISION: ([\d.]+)", l4t_line or "")
    power_mode_name = os.environ.get("JETSON_POWER_MODE_NAME")
    power_mode_id = os.environ.get("JETSON_POWER_MODE_ID")
    power_mode = (
        f"NV Power Mode: {power_mode_name}\n{power_mode_id or ''}".rstrip()
        if power_mode_name
        else _command_output(["nvpmodel", "-q"])
    )
    return {
        "architecture": platform.machine(),
        "product_model": _read_text("/proc/device-tree/model"),
        "l4t_release": (
            f"{l4t_match.group(1)}.{l4t_match.group(2)}" if l4t_match else None
        ),
        "power_mode": power_mode,
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "container_base": os.environ.get(
            "FORESIGHT_CONTAINER_BASE", "unknown (not supplied by image)"
        ),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def p(key: str, q: float) -> float | None:
        return percentile((run.get(key) for run in runs), q)

    def maximum(key: str) -> float | None:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        return max(values) if values else None

    total_duration = sum(run["e2e_latency_ms"] for run in runs) / 1000.0
    total_images = sum(int(run["image_count"]) for run in runs)
    total_output_tokens = sum(int(run["output_tokens"]) for run in runs)
    gpu_energy = sum(run.get("gpu_energy_j_estimate") or 0.0 for run in runs)
    board_energy = sum(run.get("board_energy_j_estimate") or 0.0 for run in runs)
    return {
        "ttft_ms_p50": p("ttft_ms", 0.50),
        "ttft_ms_p95": p("ttft_ms", 0.95),
        "e2e_latency_ms_p50": p("e2e_latency_ms", 0.50),
        "e2e_latency_ms_p95": p("e2e_latency_ms", 0.95),
        "measured_batches": len(runs),
        "measured_images": total_images,
        "total_inference_seconds": total_duration,
        "effective_time_per_image_ms": total_duration * 1000.0 / total_images if total_images else None,
        "image_throughput_images_s": total_images / total_duration if total_duration else None,
        "aggregate_output_throughput_tokens_s": total_output_tokens / total_duration if total_duration else None,
        "output_throughput_tokens_s_p50": p("output_throughput_tokens_s", 0.50),
        "decode_throughput_tokens_s_p50": p("decode_throughput_tokens_s", 0.50),
        "tpot_ms_p50": p("tpot_ms", 0.50),
        "torch_peak_allocated_mb": maximum("torch_peak_allocated_mb"),
        "torch_peak_reserved_mb": maximum("torch_peak_reserved_mb"),
        "system_ram_used_mb_peak": maximum("system_ram_used_mb_peak"),
        "gpu_power_w_average": gpu_energy / total_duration if gpu_energy and total_duration else None,
        "gpu_power_w_peak": maximum("gpu_power_w_peak"),
        "gpu_energy_j_estimate": gpu_energy or None,
        "board_power_w_average": board_energy / total_duration if board_energy and total_duration else None,
        "board_power_w_peak": maximum("board_power_w_peak"),
        "board_energy_j_estimate": board_energy or None,
        "temperature_c_max": maximum("temperature_c_max"),
    }


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "results.json", payload)
    batch_count = int(payload["config"]["batches"])
    images_per_batch = int(payload["config"]["images_per_batch"])
    structured_output = bool(payload["config"].get("structured_output", False))
    columns = [
        "model", "status", "ttft_ms_p50", "ttft_ms_p95", "e2e_latency_ms_p50",
        "e2e_latency_ms_p95", "measured_batches", "measured_images",
        "total_inference_seconds", "effective_time_per_image_ms",
        "image_throughput_images_s", "aggregate_output_throughput_tokens_s",
        "output_throughput_tokens_s_p50", "decode_throughput_tokens_s_p50",
        "tpot_ms_p50", "torch_peak_allocated_mb", "torch_peak_reserved_mb",
        "system_ram_used_mb_peak", "gpu_power_w_average", "gpu_power_w_peak",
        "gpu_energy_j_estimate", "board_power_w_average", "board_power_w_peak",
        "board_energy_j_estimate", "temperature_c_max", "error",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in payload["models"]:
            writer.writerow(
                {
                    "model": result["model"],
                    "status": result["status"],
                    **result.get("summary", {}),
                    "error": result.get("error", ""),
                }
            )

    lines = [
        "# vLLM Jetson benchmark", "", f"- Created: {payload['created_at']}",
        f"- Device: {payload['device'].get('product_model') or 'unknown'}",
        f"- L4T: {payload['device'].get('l4t_release') or 'unknown'}",
        f"- Power mode: `{(payload['device'].get('power_mode') or 'unknown').replace(chr(10), ' | ')}`",
        f"- Images: {len(payload['images'])} of {payload['config']['dataset_size']} at {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}",
        f"- Workload: {batch_count} batches × {images_per_batch} images per batch",
        "- Structured output: " + (
            "JSON Schema (compact JSON, backend selected by vLLM)"
            if structured_output else "disabled"
        ),
        "",
        "| Model | Status | TTFT p50/p95 (ms) | E2E p50 (ms) | Images/s | Output tok/s | Decode tok/s | Peak RAM (MB) | GPU W avg/peak | Board W avg/peak | Max °C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["models"]:
        summary = result.get("summary", {})
        if result["status"] == "ok":
            lines.append(
                "| {model} | ok | {ttft50}/{ttft95} | {e2e} | {images} | {output} | {decode} | {ram} | {gpuavg}/{gpupeak} | {boardavg}/{boardpeak} | {temp} |".format(
                    model=result["model"], ttft50=_fmt(summary.get("ttft_ms_p50")),
                    ttft95=_fmt(summary.get("ttft_ms_p95")), e2e=_fmt(summary.get("e2e_latency_ms_p50")),
                    images=_fmt(summary.get("image_throughput_images_s")),
                    output=_fmt(summary.get("aggregate_output_throughput_tokens_s")),
                    decode=_fmt(summary.get("decode_throughput_tokens_s_p50")),
                    ram=_fmt(summary.get("system_ram_used_mb_peak"), 0),
                    gpuavg=_fmt(summary.get("gpu_power_w_average")), gpupeak=_fmt(summary.get("gpu_power_w_peak")),
                    boardavg=_fmt(summary.get("board_power_w_average")), boardpeak=_fmt(summary.get("board_power_w_peak")),
                    temp=_fmt(summary.get("temperature_c_max")),
                )
            )
        else:
            lines.append(f"| {result['model']} | error |  |  |  |  |  |  |  |  |  |")
            lines.extend(["", f"Error for `{result['model']}`: `{result.get('error', 'unknown')}`"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
