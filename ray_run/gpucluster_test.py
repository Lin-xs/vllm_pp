import ray
import torch
from typing import Optional, Tuple, TYPE_CHECKING, List, Any
from functools import partial
import copy
import logging
from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# from vllm.engine.ray_utils import RayWorker
from myworker import RayWorker
from config import ParallelConfig, ModelConfig

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)



if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

def initialize_cluster(
        parallel_config: ParallelConfig,
        ray_address: Optional[str] = None
) -> Tuple[str, Optional["PlacementGroup"]]:
    ray.init(address=ray_address, ignore_reinit_error=True)
    current_placement_group = ray.util.get_current_placement_group()
    assert current_placement_group is None, "current PG is not None"
    num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
    logger.info("{} gpus detected.".format(num_gpus_in_cluster))
    if parallel_config.world_size > num_gpus_in_cluster:
        raise ValueError(
            "The number of required GPUs exceeds the total number of "
            "available GPUs in the cluster.")
    # Create a new placement group
    current_placement_group = ray.util.placement_group([{
        "GPU": 1
    }] * parallel_config.world_size)
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    ray.get(current_placement_group.ready(), timeout=1800)
    return None, current_placement_group

class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        # cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        # scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        self.model_config = model_config
        # self.cache_config = cache_config
        # assert self.cache_config.sliding_window == getattr(
        #     self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        # self.scheduler_config = scheduler_config

        self._init_workers_ray(placement_group)
    def _init_workers_ray(self, placement_group: "PlacementGroup",
                            **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        # from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel
        from myworker import Worker

        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote()
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        # scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker",
                            get_all_outputs=True,
                            worker_init_fn=lambda: Worker(
                                model_config,
                                parallel_config,
                                #scheduler_config,
                                None,
                                None,
                                None,
                            ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

if __name__ == '__main__':
    parallel_config = ParallelConfig(4, 1, False)
    model_config = ModelConfig()

    distributed_init_method, placement_group = initialize_cluster(parallel_config)
    engine = LLMEngine(model_config=model_config,
                       parallel_config=parallel_config,
                       distributed_init_method=distributed_init_method,
                       placement_group=placement_group,
                       log_stats=False)
    res = engine._run_workers("excute_model", get_all_outputs=True)
    print(res)