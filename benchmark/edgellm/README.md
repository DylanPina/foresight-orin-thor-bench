# TensorRT Edge-LLM Benchmarks

Runs the same workload as [`benchmark/vllm`](../vllm/README.md) (40 COCO
images at 336×224, 10 batches × 4 images, 512 max tokens, greedy, thinking off)
through NVIDIA's [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
C++ runtime so the two runtimes can be compared row for row.

Only Jetson Thor (JetPack 7.0/7.1, CUDA 13.0) is covered. Edge-LLM supports
Orin only from JetPack 7.2, which the Orin vLLM image (JetPack 6) does not match.

## Run the container

```bash
# Jetson Thor
./container build --name edgellm-thor --dockerfile docker/Dockerfile.thor_edgellm
./container shell --name edgellm-thor --dockerfile docker/Dockerfile.thor_edgellm
```

The image starts from `nvcr.io/nvidia/tensorrt:25.11-py3` (CUDA 13.0.2,
TensorRT 10.14.1), compiles Edge-LLM `v0.10.1` for `jetson-thor` under
`/opt/TensorRT-Edge-LLM`, and installs the CPU-only ONNX export environment in
`/opt/edgellm-venv`. Both are on `PATH`.

## Run the benchmark

```bash
cd /home/ros/foresight_ws

python3 -m benchmark.edgellm.benchmark
```

Leave `HF_HUB_OFFLINE` unset for the first export of each model. The exporter
uses `snapshot_download`, which in offline mode insists on a complete snapshot,
and the vLLM runs only ever fetched the weights and configs (`.gitattributes`,
`LICENSE`, and `README.md` are missing). Once the ONNX export is cached the
benchmark no longer touches the Hub, so `HF_HUB_OFFLINE=1` is safe afterwards.

With no `--models` argument, the benchmark runs the same four models as the
vLLM benchmark, all in FP16:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `google/gemma-4-E2B-it` (exported with `--skip-audio`)
- `ut-amrl/foresight-qwen3vl-2b-sft` (Qwen3-VL family)

For each model the worker:

1. exports the checkpoint to ONNX (`tensorrt-edgellm-export`, CPU),
2. builds the LLM and visual TensorRT engines (`llm_build`, `visual_build`),
3. runs one `llm_inference` invocation per image batch with `--warmup` and
   `--profileOutputFile`, and maps the profile onto the vLLM result schema.

ONNX exports and engines are cached under `data/edgellm/<model-slug>/{onnx,engines}`
and reused on later runs. Engines are rebuilt automatically when the build
parameters change (they are recorded in `engines/foresight_build_params.json`).
Pass `--rebuild-engines` to force both steps, or `--skip-export` to fail
instead of exporting when a cache is missing (for example when ONNX files were
produced on another machine).

Exports and engine builds land in a staging directory and are renamed into
place only when complete, and each model's workspace is locked (`.foresight.lock`)
for the duration of its export, build, and measurement. A second benchmark
process that targets the same model — for example from another container —
waits for the first to finish instead of rebuilding engines underneath it.
Even so, do not run two benchmarks at once: they share the GPU and both sets
of timings would be contaminated.

Per-family settings live in `MODEL_FAMILY_PROFILES` in `config.py`. Gemma 4
pins `--maxImageTokensPerImage 280` because the Edge-LLM runner takes its
per-image soft-token cap from the visual engine rather than from the
processor's `max_soft_tokens`; 280 is what the vLLM benchmark used.

Results are written to `data/benchmarks/edgellm/<timestamp>/` with the same
`results.json`, `results.csv`, and `summary.md` layout as the vLLM benchmark,
plus per-model `build.log`, `inference.log`, and the raw request/output/profile
JSON under `models/<model-slug>/requests/`.

### Configuration

```bash
python3 -m benchmark.edgellm.benchmark \
  --models Qwen/Qwen3.5-0.8B \
  --batches 1 \
  --images-per-batch 4 \
  --runs 1
```

`--max-model-len` (default 2560) becomes the engine's `--maxKVCacheCapacity`
and `--max-input-len` (default 2048) its `--maxInputLen`; the visual engine is
built for `images-per-batch × --max-image-tokens-per-image` (default 512)
image tokens. Use `--help` to list all options.

## Comparing with vLLM

See the [top-level README](../README.md) for `python3 -m benchmark.compare`.

## Metric mapping

| Result field | vLLM source | Edge-LLM source (`--profileOutputFile`) |
|---|---|---|
| `e2e_latency_ms` | wall time around `llm.generate` | `wall_clock.total_time_ms` (post-warmup, excludes engine load) |
| `ttft_ms` | `first_token_latency` | vision-encoder stage GPU ms + `prefill.average_time_per_run_ms` (estimate) |
| `output_tokens` | completion token count | `wall_clock.generated_tokens` |
| `decode_throughput_tokens_s` | decode timestamps | `generation.tokens_per_second` |
| `tpot_ms` | decode timestamps | `generation.average_time_per_token_ms` |
| `torch_peak_*_mb` | torch allocator | n/a (`runtime_peak_gpu_memory_mb` from `memory`) |
| power / thermal | tegrastats during the request | tegrastats from the `Processing … batched requests` log line onward |

## Known differences

- Edge-LLM runs FP16 engines; the vLLM runs use the checkpoints' native BF16.
- Edge-LLM has no JSON Schema constrained decoding, so `--structured-output`
  does not exist here (vLLM's default is disabled, so default runs match).
- Each Edge-LLM batch is a separate process; engine load time is excluded from
  every metric, whereas vLLM keeps one engine resident per model.
- The runtime's encoder-embedding cache is disabled (`--encoderCacheBudgetBytes 0`)
  so the warmup pass cannot serve the measured pass's vision embeddings,
  mirroring vLLM's `mm_processor_cache_gb=0` and disabled prefix caching.
- Edge-LLM TTFT is a GPU-stage estimate, not a first-token timestamp. Treat
  E2E latency and decode tokens/s as the primary comparison metrics.
