# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import OPTConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_pipeline_model_parallel_first_rank, get_pipeline_model_parallel_last_rank, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank, get_pipeline_model_parallel_next_rank, get_pipeline_model_parallel_prev_rank,
    is_first_pipeline_stage, is_last_pipeline_stage)
from vllm.model_executor.parallel_utils.layers import (VocabParallelEmbedding,
                                                       ColumnParallelLinear,
                                                       RowParallelLinear)
from vllm.sequence import SamplerOutput
from vllm.model_executor.parallel_utils.utils import send_metadata, send_tensor_and_shape, recv_metadata, recv_tensor

KVCache = Tuple[torch.Tensor, torch.Tensor]
import copy


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            embed_dim,
            3 * embed_dim,
            bias=bias,
            gather_output=False,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            input_is_parallel=True,
        )
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scale=self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata, cache_event)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            gather_output=False,
        )
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            input_is_parallel=True,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       input_metadata=input_metadata,
                                       cache_event=cache_event)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        if is_first_pipeline_stage() or is_last_pipeline_stage():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.word_embed_proj_dim,
            )
            # Positional embeddings are replicated (not sharded).
            self.embed_positions = OPTLearnedPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_tokens = None
            self.embed_positions = None

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size,
                                         config.word_embed_proj_dim,
                                         bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim,
                                        config.hidden_size,
                                        bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None

        assert config.num_hidden_layers % get_pipeline_model_parallel_world_size() == 0
        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers // get_pipeline_model_parallel_world_size())])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
        metadata_stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        use_pipeline = get_pipeline_model_parallel_world_size() > 1
        if is_first_pipeline_stage():
            # this is the first in pipeline, or No pipeline.
            inputs_embeds = self.embed_tokens(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        elif use_pipeline:
            # Use pipeline, but not the first
            input_metadata = recv_metadata()
            hidden_states = recv_tensor(torch.cuda.current_device(), src=get_pipeline_model_parallel_prev_rank())

        #  TODO: wyq add
        # from vllm.utils import ctx_get_inteval_datetime
        # with ctx_get_inteval_datetime("Transformer blocks"):
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                cache_event)
            if use_pipeline and is_first_pipeline_stage() and i == 0:
                with torch.cuda.stream(metadata_stream):
                    send_metadata(input_metadata=input_metadata)
        
        print(f"rank {get_pipeline_model_parallel_rank()}: layer cal over. shape: {hidden_states.shape}")
        
        # print(f"shape of hidden_states: {hidden_states.shape}")
        if is_last_pipeline_stage():
            # this is the last rank in pipeline, or No pipeline
            if self.final_layer_norm is not None:
                hidden_states = self.final_layer_norm(hidden_states)
            if self.project_out is not None:
                hidden_states = self.project_out(hidden_states)
        else:
            # if is_first_pipeline_stage():
            #     metadata_stream.synchronize()
            try:
                send_tensor_and_shape(hidden_states, get_pipeline_model_parallel_next_rank())
            except Exception as e:
                print("error rank = {}".format(get_pipeline_model_parallel_rank()))
                raise e
        return hidden_states


class OPTModel(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.decoder = OPTDecoder(config)
        self.metadata_stream = torch.cuda.Stream()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, kv_caches, input_metadata,
                            cache_events, self.metadata_stream)


class OPTForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = OPTModel(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        if is_last_pipeline_stage():
            self.lm_head_weight = self.model.decoder.embed_tokens.weight
        else:
            self.lm_head_weight = None
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        # TODO: Add by wyq
        # from vllm.utils import ctx_get_inteval_datetime
        # with ctx_get_inteval_datetime("Original OPTForCausalLM"):
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = None
        if is_last_pipeline_stage():
            next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                    input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "fc1.weight", "fc1.bias"
    ]
    _row_parallel_weights = ["out_proj.weight", "fc2.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()
        state_dict = self.state_dict()
        num_layer_per_stage = self.config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        
        def is_layer_in_this_stage(layer_id: int):
            return num_layer_per_stage * pipeline_model_parallel_rank <= layer_id < num_layer_per_stage * (pipeline_model_parallel_rank + 1)
        
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "lm_head.weight" in name:
                continue

            if name.startswith("decoder."):
                name = "model." + name

            layer_id = -1
            str_layer, str_id = name.split(".")[2:4]
            if str_layer == "layers":
                layer_id = int(str_id)
            else:
                # 这条分支的权重是：
                # model.decoder.embed_tokens.weight
                # model.decoder.embed_positions.weight
                # model.decoder.final_layer_norm.weight
                # model.decoder.final_layer_norm.bias
                if not(is_first_pipeline_stage() or is_last_pipeline_stage()):
                    continue
                param = state_dict[name]
                load_tensor_parallel_weights(param, loaded_weight, name,
                                            self._column_parallel_weights,
                                            self._row_parallel_weights,
                                            tensor_model_parallel_rank)
                
            if not is_layer_in_this_stage(layer_id):
                continue
            local_layer_id: int = layer_id % num_layer_per_stage
                
            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj").replace(str_id, str(local_layer_id))]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            param = state_dict[name.replace(str_id, str(local_layer_id))]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
