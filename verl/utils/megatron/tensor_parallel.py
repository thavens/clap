# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch.nn import init

if TYPE_CHECKING:
    from megatron.core import ModelParallelConfig


def update_kwargs_with_config(dictionary: dict, config: "ModelParallelConfig"):
    dictionary["config"] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_tensor_parallel_partition_dim(param):
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_entropy_with_chunking(vocab_parallel_logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Memory-efficient entropy calculation using chunked processing when logits are sharded in tp ranks.
    Args:
        vocab_parallel_logits: (batch_size, seq_len, vocab_size // tp_size) or (total_nnz, vocab_size // tp_size)
        chunk_size: Number of sequence tokens to process at once. Defaults to 2048.
    Returns: (batch_size, seq_len)
    """

    output_shape = list(vocab_parallel_logits.shape[:-1])
    entropy = torch.zeros(output_shape, device=vocab_parallel_logits.device)

    for i in range(0, vocab_parallel_logits.shape[1], chunk_size):
        logits_chunk = vocab_parallel_logits[:, i : i + chunk_size, :]
        entropy_chunk = _VocabParallelEntropy.apply(logits_chunk)
        entropy[:, i : i + chunk_size] = entropy_chunk

    return entropy


def _vocab_parallel_shifted_exp(vocab_parallel_logits: torch.Tensor, group) -> tuple[torch.Tensor, ...]:
    """Max-shift vocab-parallel logits and tp-reduce the softmax denominator.

    Returns ``(shifted, exp_shifted, sum_exp)``, where ``shifted`` is the local shard minus
    the *global* max and ``sum_exp`` is the *global* ``sum(exp(shifted))`` over the full
    vocab. Non-destructive, so callers can still consume ``vocab_parallel_logits`` after.
    """
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
    shifted = vocab_parallel_logits - logits_max
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=group)
    return shifted, exp_shifted, sum_exp


def vocab_parallel_sum_pi_squared(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute Σπ² (sum of squared probabilities) when logits are sharded across tp ranks.

    Used by ``optimal_token_baseline`` advantage estimators as the path-variance proxy:
    ``w_t = 1 - 2*π_t + Σπ²``.

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size)

    Returns: (...,)

    Implementation is non-destructive (does not mutate ``vocab_parallel_logits``) so it
    can be safely called before ``vocab_parallel_entropy`` / ``vocab_parallel_log_probs``
    which would otherwise consume the same tensor.
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    _, exp_shifted, sum_exp = _vocab_parallel_shifted_exp(vocab_parallel_logits, tp_group)
    sum_exp_squared = exp_shifted.pow(2).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp_squared, group=tp_group)
    return (sum_exp_squared / sum_exp.pow(2)).squeeze(dim=-1)


def vocab_parallel_log_probs_from_logits(logits, labels):
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel
    region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


def vocab_parallel_log_softmax_reduced(vp_logits: torch.Tensor, group=None) -> torch.Tensor:
    """TP-reduced log-softmax over vocab-parallel logits (local shard in, local shard out).

    Args:
        vp_logits: (..., vocab_size // tp_size) local-shard logits.
        group: process group holding the vocab shards; defaults to the megatron TP group.
    """
    if group is None:
        group = mpu.get_tensor_model_parallel_group()

    shifted, _, sum_exp = _vocab_parallel_shifted_exp(vp_logits.float(), group)
    return shifted - sum_exp.log()


def vocab_parallel_topk_log_probs(vp_logits: torch.Tensor, k: int, *, group=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the global top-k log-probs and ids from vocab-parallel logits.

    Each TP rank holds a vocab shard of ``vp_logits``; this returns the same global
    top-k on every rank. Used by the SDPO self-teacher forward to emit ``teacher_ids``
    / ``teacher_logprobs`` (the model's own top-k), the transport the SDPO loss consumes.

    Args:
        vp_logits: (..., vocab_size // tp_size) local-shard logits (temperature already applied).
        k: number of top candidates.
        group: process group holding the vocab shards; defaults to the megatron TP group.
            Overridable so the routine can be exercised on a plain process group in tests.

    Returns:
        (topk_log_probs, topk_ids) each (..., k), with ids in global vocab coordinates.
    """
    if group is None:
        group = mpu.get_tensor_model_parallel_group()
    rank, world_size = dist.get_rank(group=group), dist.get_world_size(group=group)

    # Megatron shards the vocab into equal contiguous blocks (see VocabUtility), so this
    # rank's shard covers [vocab_start_index, vocab_start_index + partition_vocab_size).
    partition_vocab_size = vp_logits.size(-1)
    vocab_start_index = rank * partition_vocab_size

    vp_log_probs = vocab_parallel_log_softmax_reduced(vp_logits, group=group)

    local_k = min(k, partition_vocab_size)
    local_topk_log_probs, local_topk_ids = torch.topk(vp_log_probs, k=local_k, dim=-1)
    local_topk_ids = local_topk_ids + vocab_start_index

    gathered_log_probs = [torch.empty_like(local_topk_log_probs) for _ in range(world_size)]
    gathered_ids = [torch.empty_like(local_topk_ids) for _ in range(world_size)]
    dist.all_gather(gathered_log_probs, local_topk_log_probs, group=group)
    dist.all_gather(gathered_ids, local_topk_ids, group=group)

    # The global top-k is the top-k of the concatenated per-shard top-k candidates.
    candidate_log_probs = torch.cat(gathered_log_probs, dim=-1)
    candidate_ids = torch.cat(gathered_ids, dim=-1)
    topk_log_probs, positions = torch.topk(candidate_log_probs, k=k, dim=-1)
    return topk_log_probs, torch.gather(candidate_ids, dim=-1, index=positions)
