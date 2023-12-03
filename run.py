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

sampling_params = SamplingParams(temperature=0.8,
                                    top_p=0.95,
                                    max_tokens=4)

llm = LLM(model="facebook/opt-6.7b")