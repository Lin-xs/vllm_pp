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
        default="facebook/opt-6.7b",
        choices=["facebook/opt-66b", "facebook/opt-13b", "facebook/opt-6.7b", "facebook/opt-125m", "facebook/opt-2.7b", "facebook/opt-30b"]
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
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=10
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256
    )
    parser.add_argument(
        "--minibatch_chunk",
        type=int,
        default=1
    )
    return parser.parse_args()

def write_log(text: str):
    log_file = "/home/v-yuanqwang/vllm_pp/nsys-rep/pp_time_record/rank_{}.log"
    for i in range(4):
        ff = log_file.format(i)
        with open(ff, "a") as f:
            f.write(text)



prompts = []
with open("./nsys-rep/prompts/inclen_english_30.txt", "r") as f:
    for line in f:
        prompts.append(line)
# prompts = prompts * 500
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
                max_num_seqs = args.max_num_seqs,
                disable_log_stats=False
                )
    else:
        llm = LLM(model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs = args.max_num_seqs,
                minibatch_chunk = args.minibatch_chunk,
                max_num_batched_tokens = 3000,
                )
    # with ctx_get_inteval_datetime("RUN ALL"):
        
    # prompts_source = "./nsys-rep/prompts/longprompts-{}.txt"
    # for plen in [100, 200, 300, 400, 500, 600]:
    #     prompts = []
    #     with open(prompts_source.format(plen), "r") as f:
    #         for line in f:
    #             prompts.append(line)
    #     prompts = prompts * 500

    #     for bs in range(1, 100):
    #         write_log("[")
    #         outputs = llm.generate(prompts[:bs], sampling_params)
    #         write_log("]\n")
    #     write_log("\n")

    # prompts = ["La maladie"] * 500
    # prompts = ["La maladie de Creutzfeldt-Jakob (MCJ) est une affection"] * 800
    # batch_ls = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
    # for bs in range(8, 512+1, 8):
    #     write_log("[")
    #     outputs = llm.generate(prompts[:bs], sampling_params)
    #     write_log("]\n")
    
    for _ in range(2):
        write_log("[")
        outputs = llm.generate(prompts, sampling_params)
        write_log("]\n")


    # for output in outputs[:1]:
    #         prompt = output.prompt
    #         generated_text = output.outputs[0].text
    #         print(f"\nPrompt: {prompt!r} \nGenerated text: {generated_text!r}")
        
