from vllm import LLM, SamplingParams
try:
    from vllm.engine.llm_engine import exit_pipeline
except ImportError:
    pass
try:
    from vllm.utils import ctx_get_inteval_datetime
except ImportError:
    pass
import time
import argparse
import torch

def send_test():
    import torch
    import torch.distributed as dist
    if dist.is_initialized():
        with ctx_get_inteval_datetime("Send Test", sync=True, device=0):
            tensor = torch.randn(100, 100).to(0)
            dist.send(tensor, dst=1)
        with ctx_get_inteval_datetime("Recv Test", sync=True, device=0):
            dist.recv(tensor, src=1)

def _parse_args():
    parser = argparse.ArgumentParser("vLLM model run test")
    parser.add_argument(
        "--model",
        default="facebook/opt-6.7b",
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
        "--pipeline_parallel_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90
    )
    return parser.parse_args()
    
prompts = [
    "What is considered a high blood pressure reading?",
    "What are the risks of having high blood pressure?",
    "What is bronchiolitis and who is most susceptible to it?",
    "What treatments are available for bronchiolitis?",
    "What is the difference between bronchiolitis and bronchitis?",
    "What are the symptoms of Creutzfeldt-Jakob disease?",
    "What is the Variant CJD compensation scheme?",
    "How is HIV transmitted from one person to another?",
    "What are some of the ways to diagnose HIV?",
    "What causes laryngitis besides cold or flu infection?"
]

if __name__ == "__main__":
    args = _parse_args()
    max_token = args.max_tokens

    sampling_params = SamplingParams(temperature=0.8,
                                    top_p=0.95,
                                    max_tokens=max_token)

    llm = LLM(model=args.model,
              pipeline_parallel_size=args.pipeline_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization
            )

    if args.pipeline:
        print("PIPELINE ENABLED")
        # send_test()


    # st = time.time()
    # with ctx_get_inteval_datetime("Warm Up"):
    #     outputs = llm.generate(prompts[:1], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2))
    # ed = time.time()
    # print(f"Time cost_1: {ed-st:.4f} s")

    time.sleep(0.3)

    batch_size = [int(bs) for bs in args.batch_sizes_list.split(",")]
    for i, bs in enumerate(batch_size):
        st = time.time()
        with ctx_get_inteval_datetime(f"{i} th request for batch size = {bs}", sync=True):
            outputs = llm.generate(prompts[1:1+bs], sampling_params)
        ed = time.time()
        print(f"max tokens: {max_token}. Time cost for batch size = {bs} : {ed-st:.4f} s")
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt!r} \nGenerated text: {generated_text!r}")

    print("Memory footprint:{:.2f} GB".format(torch.cuda.max_memory_allocated()/2**30))
    try:
        # exit_pipeline(broadcast=False)
        exit_pipeline()
    except NameError as e:
        pass

