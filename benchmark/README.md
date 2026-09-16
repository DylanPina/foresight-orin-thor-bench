# Jetson VLM benchmarks

Bench marking VLMs on vLLM vs. Tensor RT Edge-LLM. 
The workload consists of 40 COCO images at 336x224, 10 batches x 4 images, a fixed
object-enumeration prompt, 512 max tokens, greedy decoding, with thinking off.

| Package | Runtime | Container | Docs |
|---|---|---|---|
| `benchmark.vllm` | vLLM | `docker/Dockerfile.thor_vllm`, `docker/Dockerfile.orin_vllm` | [vllm/README.md](vllm/README.md) |
| `benchmark.edgellm` | TensorRT Edge-LLM | `docker/Dockerfile.thor_edgellm` | [edgellm/README.md](edgellm/README.md) |

Both run from the repository root inside their container:

```bash
python3 -m benchmark.vllm.benchmark
python3 -m benchmark.edgellm.benchmark
```

and write `results.json`, `results.csv`, and `summary.md` with the same schema
to `data/benchmarks/<runtime>/<timestamp>/`.

## Layout

- `common/config.py` — the workload (models, images, prompt, batch limits) and
  the command-line options every runtime shares.
- `common/utils.py` — tegrastats/RAM telemetry (`ResourceMonitor`), image
  preparation, Jetson device metadata and memory reclamation, percentile
  statistics, and the JSON/CSV/Markdown report writers.
- `common/runner.py` — the coordinator/worker loop: one worker process per
  model, results collected into a single report.
- `vllm/`, `edgellm/` — runtime-specific configuration, engine setup, and the
  per-model `run_model` implementation.
- `compare.py` — joins a vLLM run and an Edge-LLM run into one comparison.

## Comparing runs

```bash
python3 -m benchmark.compare \
  --vllm data/benchmarks/vllm/20260916T183609Z \
  --edgellm data/benchmarks/edgellm/20260916T182042Z
```

writes `comparison.md` / `comparison.csv` under
`data/benchmarks/comparison/<timestamp>/` with per-model speedups. Speedup is
vLLM ÷ Edge-LLM for latency and energy and Edge-LLM ÷ vLLM for throughput, so
`> 1` always means Edge-LLM did better. Edge-LLM's TTFT is a GPU-stage estimate
(vision encoder + prefill) rather than a measured first-token timestamp, so
treat E2E latency and decode tokens/s as the primary comparison metrics.
