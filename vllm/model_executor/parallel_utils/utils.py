# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from typing import List, Sequence, TYPE_CHECKING

import torch
import torch.distributed as dist
import pickle

from vllm.model_executor.parallel_utils.parallel_state import (get_pipeline_model_parallel_first_rank,
                                                               get_pipeline_model_parallel_next_rank,
                                                               get_pipeline_model_parallel_prev_rank,
                                                               get_pipeline_model_parallel_last_rank,
                                                               get_pipeline_model_parallel_world_size,
                                                               get_pipeline_model_parallel_rank)
if TYPE_CHECKING:
    from vllm.model_executor.input_metadata import InputMetadata

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class VocabUtility:
    """ Split the vocabulary into `world_size` chunks and return the first
        and last index of the vocabulary belonging to the `rank`
        partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size: int, rank: int) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int,
                                           world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank)

def send_tensor_and_shape(tensor: torch.Tensor, dst: int):
    a = torch.tensor(tensor.shape, dtype=torch.int).to(tensor.device)
    #_print_log(f"send shape tensor: {a}")
    assert a.shape[0] == 2, "dist send: tensor dim != 2"

    dist.send(a, dst)
    dist.send(tensor, dst)
    
    #_print_log(f"Send tensor shape:{tensor.shape}")


def recv_tensor(dev, src: int | None) -> torch.Tensor:
    shape = torch.zeros(2, dtype=torch.int32, device=dev)
    dist.recv(shape, src)
    #_print_log(f"recv shape tensor: {shape}")
    tensor = torch.zeros(list(shape), dtype=torch.float16, device=dev)
    dist.recv(tensor, src)
    return tensor
    
def send_metadata(input_metadata: "InputMetadata", broadcast = True):
    serialized_instance = pickle.dumps(input_metadata)
    size_tensor = torch.tensor([len(serialized_instance)], dtype=torch.long).cuda()
    if broadcast:
        dist.broadcast(size_tensor, src=get_pipeline_model_parallel_first_rank())
    else:
        dist.send(size_tensor, dst=get_pipeline_model_parallel_next_rank())
    data_tensor = torch.tensor(list(serialized_instance), dtype=torch.uint8).cuda()
    
    if broadcast:
        dist.broadcast(data_tensor, src=get_pipeline_model_parallel_first_rank())
    else:
        dist.send(data_tensor, dst=get_pipeline_model_parallel_next_rank())

def recv_metadata(broadcast = True) -> "InputMetadata":
    # with ctx_get_inteval_datetime("Recv Meta Size", sync=True):
    size_tensor = torch.zeros(1, dtype=torch.long).cuda()
    if broadcast:
        dist.broadcast(size_tensor, src=get_pipeline_model_parallel_first_rank())
    else:
        dist.recv(size_tensor, src=get_pipeline_model_parallel_prev_rank())
    data_size = size_tensor.item()
    print("Byte data size:{}".format(data_size))
    if data_size == 0:
        print("EXIT PIPELINE.")
        if get_pipeline_model_parallel_rank != get_pipeline_model_parallel_last_rank():
            dist.send(size_tensor, dst=get_pipeline_model_parallel_next_rank())
        raise KeyboardInterrupt

    # with ctx_get_inteval_datetime("Recv Metadata", sync=True):
    # 接收实际数据
    data_tensor = torch.zeros(data_size, dtype=torch.uint8).cuda()
    
    if broadcast:
        dist.broadcast(data_tensor, src=get_pipeline_model_parallel_first_rank())
    else:
        dist.recv(data_tensor, src=get_pipeline_model_parallel_prev_rank())
    # print("recv data_tensor: {}".format(data_tensor))
    input_metadata: "InputMetadata" = pickle.loads(bytes(data_tensor.tolist()))
    input_metadata.slot_mapping = input_metadata.slot_mapping.cuda()
    input_metadata.context_lens = input_metadata.context_lens.cuda()
    input_metadata.block_tables = input_metadata.block_tables.cuda()
    if input_metadata.to_cache is not None:
        input_metadata.to_cache = input_metadata.to_cache.cuda()
    return input_metadata