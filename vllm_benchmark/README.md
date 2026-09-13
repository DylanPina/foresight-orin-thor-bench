# vLLM Benchmarks

## Run the container

From the repository root on the Jetson host, build and start the container for
your device:

```bash
# Jetson Orin
./container build --name vllm-orin --dockerfile docker/Dockerfile.orin_vllm
./container shell --name vllm-orin --dockerfile docker/Dockerfile.orin_vllm

# Jetson Thor
./container build --name vllm-thor --dockerfile docker/Dockerfile.thor_vllm
./container shell --name vllm-thor --dockerfile docker/Dockerfile.thor_vllm
```

The `build` command also starts the container. Later, use `shell` to reopen it
or `stop` to stop it:

```bash
./container stop --name vllm-orin
./container stop --name vllm-thor
```

The Orin image retains its JetPack 6 / CUDA 12.6 build of vLLM 0.19.0 and
overlays Transformers 5.5.3. Gemma 4 support in this vLLM release requires the
`Gemma4Processor` that is absent from Transformers 4.x.

## Run the benchmark

Inside the container, run the benchmark from the mounted repository root:

```bash
cd /home/ros/argo_ws

# After the models have been downloaded from Hugging Face, prevent network
# requests and require locally cached model files.
export HF_HUB_OFFLINE=1

python3 -m vllm_benchmark.benchmark
```

With no `--models` argument, the benchmark runs these four models:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `google/gemma-4-E2B-it`
- `ut-amrl/foresight-qwen3vl-2b-sft`

### Configuration

```bash
python3 -m vllm_benchmark.benchmark \
  --models ut-amrl/foresight-qwen3vl-2b-sft \
  --batches 1 \
  --images-per-batch 4 \
  --runs 1
```

Use `--help` to list all options. Results are written to
`data/benchmarks/vllm/<timestamp>/` as JSON, CSV, logs, and a Markdown summary.
