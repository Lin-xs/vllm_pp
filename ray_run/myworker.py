import ray
from ray.air.util.torch_dist import TorchDistributedWorker
import os
import torch
import torch.distributed
import torch.distributed as dist
from typing import Optional
from config import ParallelConfig, set_random_seed, ModelConfig
from parallel_state import initialize_model_parallel
import parallel_state

def get_model(config: ModelConfig):
    return torch.nn.Linear(config.input_size, config.output_size).cuda()

class RayWorker(TorchDistributedWorker):
    """Ray wrapper for vllm.worker.Worker, allowing Worker to be
    lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

    def __init__(self, init_cached_hf_modules=False) -> None:
        if init_cached_hf_modules:
            # pylint: disable=import-outside-toplevel
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        self.worker = None

    def init_worker(self, worker_init_fn):
        self.worker = worker_init_fn()

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def execute_method(self, method, *args, **kwargs):
        executor = getattr(self, method)
        return executor(*args, **kwargs)
    

class Worker:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config = None,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.parallel_config = parallel_config
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)


        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)

    def excute_model(self, input: torch.Tensor = None):
        if parallel_state.get_pipeline_model_parallel_last_rank() == parallel_state.get_pipeline_model_parallel_rank():
            print("this is the last rank. rank = {}".format(torch.distributed.get_rank()))
        
        if parallel_state.get_pipeline_model_parallel_first_rank() == parallel_state.get_pipeline_model_parallel_rank():
            print("this is the last rank. rank = {}".format(torch.distributed.get_rank()))
            tensor = torch.randn(1000, 1000).cuda()
            output = self.model(tensor)
            dist.send(output, parallel_state.get_pipeline_model_parallel_next_rank())
            print("rank0 ok")
            return
        recv_tensor = torch.empty(1000, 1000).cuda()
        print("prevrank = {}".format(parallel_state.get_pipeline_model_parallel_prev_rank()))
        dist.recv(recv_tensor, parallel_state.get_pipeline_model_parallel_prev_rank())
        output = self.model(recv_tensor)
        if parallel_state.get_pipeline_model_parallel_last_rank() == parallel_state.get_pipeline_model_parallel_rank():
            print(output.sum())
            return
        dist.send(output, parallel_state.get_pipeline_model_parallel_next_rank())
        



def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)