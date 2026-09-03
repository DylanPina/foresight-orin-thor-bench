# -*- coding: utf-8 -*-
import os
import time

# Set memory-sensitive env vars before importing torch/vllm.
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ["VLLM_USE_V1"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """
You are a helpful navigation robot that can understand multi-view images and language. You are tasked with navigating to the goal location given by a language instruction. You will be shown a sequence of first person images in chronological order and asked to generate a realistic motion plan and critique of the motion plan. 

You will be asked these questions in a conversational format and must reflect on your past responses to revise your motion plans. The output format, requirements of each role, and annotated motion plan will be provided by the user.
"""
MOTION_PROMPT = """
Sample a unique motion trajectory from the distribution of trajectories that follows the language instruction:
(turn left after crossing the wooden bridge)
while satisfying the following constraints.

Constraints:
- The trajectory must be in normalized pixel coordinates. Each point is [x,y] with 0<=x<=1, 0<=y<=1.
- Use <= 10 points. The first point MUST be at (0.5, 1.0) as indicated by the red dot on the image.
- Goal: the path must accurately follow the language command
  - Do not plan points above the ground horizon line, this varies by image.
- Follow the language instruction while complying with the other constraints.
- Smooth and plausible motion (no sharp zig-zags).

Self-check before output:
- Verify the final path demonstrates safe behavior that precisely follows the language command.
If not, adjust the points towards goal instruction.

Output ONLY JSON with exactly one key "trajectory". No extra text. 
Format:
{"trajectory":[[x0, y0], ... [xn, yn]]}
"""

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
              {
                  "type": "image",
                  "image": Image.open("grandtour__mission_2024-11-15-14-14-12__2675.png")
              },
              {"type": "text", "text": MOTION_PROMPT},
            ],
        }
    ]

    # TODO: change to your own checkpoint path
    # checkpoint_path = "checkpoints/Qwen3-VL-2B-Instruct"
    checkpoint_path = "checkpoints/Qwen3-VL-2B-Instruct-AWQ-8bit"
    # Cap visual tokens for faster first-token latency on Jetson.
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,
    )
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]
    tp_size = 1

    llm = LLM(
        model=checkpoint_path,
        # Orin-safe defaults to reduce init-time GPU memory pressure.
        gpu_memory_utilization=0.75,
        enforce_eager=False,
        max_num_seqs=1,
        tensor_parallel_size=tp_size,
        max_model_len=1024,
        disable_log_stats=True,
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated text: {generated_text!r}")

    # Benchmark: run the same prompt 10 times after first pass.
    num_trials = 10
    latencies_s = []
    token_counts = []
    for trial in range(num_trials):
        t0 = time.perf_counter()
        trial_outputs = llm.generate(inputs, sampling_params=sampling_params)
        dt = time.perf_counter() - t0
        latencies_s.append(dt)

        # Sum output token counts across responses in this batch.
        tokens_this_trial = 0
        for out in trial_outputs:
            if out.outputs:
                token_ids = getattr(out.outputs[0], "token_ids", None)
                if token_ids is not None:
                    tokens_this_trial += len(token_ids)
                else:
                    # Fallback if token_ids are unavailable.
                    tokens_this_trial += len(out.outputs[0].text.split())
        token_counts.append(tokens_this_trial)
        print(f"Trial {trial + 1}/{num_trials}: {dt:.3f}s, tokens={tokens_this_trial}")

    min_t = min(latencies_s)
    max_t = max(latencies_s)
    avg_t = sum(latencies_s) / len(latencies_s)
    total_tokens = sum(token_counts)
    avg_s_per_token = (sum(latencies_s) / total_tokens) if total_tokens > 0 else float("inf")

    print()
    print("=" * 40)
    print("Benchmark summary (10 repeated inferences)")
    print(f"Min inference time: {min_t:.3f} s")
    print(f"Max inference time: {max_t:.3f} s")
    print(f"Avg inference time: {avg_t:.3f} s")
    print(f"Avg time per token: {avg_s_per_token:.6f} s/token")
