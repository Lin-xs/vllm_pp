from vllm import LLM, SamplingParams
try:
    from vllm.utils import ctx_get_inteval_datetime
except ImportError:
    pass
import time
import argparse
import torch

def _parse_args():
    parser = argparse.ArgumentParser("vLLM model run test")
    parser.add_argument(
        "--model",
        default="facebook/opt-30b",
        choices=["facebook/opt-6.7b", "facebook/opt-125m", "facebook/opt-2.7b"]
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True
    )
    parser.add_argument(
        "--pipeline",
        action="store_true"
    )
    parser.add_argument(
        "--batch_sizes_list",       # --batch_sizes_list 1,2,4,8
        "-bsl",
        type=str,
        default="1,2,4,8",
        help="A list of batch size. e.g.: --batch_sizes_list 1,2,4,8"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90
    )
    parser.add_argument(
         "--verbose",
         "-v",
         action="store_true"
    )
    return parser.parse_args()

prompts = []
with open("./prompts.txt", "r") as f:
    for line in f:
        prompts.append(line)

# prompts = prompts[:10]

if __name__ == "__main__":
    args = _parse_args()
    max_token = args.max_tokens

    sampling_params = SamplingParams(temperature=0.8,
                                    top_p=0.95,
                                    max_tokens=max_token)
    if args.verbose:
        llm = LLM(model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                disable_log_stats=False
                )
    else:
        llm = LLM(model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                )
    with ctx_get_inteval_datetime("RUN ALL"):
        outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt!r} \nGenerated text: {generated_text!r}")
        
