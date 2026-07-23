# Copyright 2026 Individual Contributor: Michael Lavery
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

"""Real vocab-parallel validation of the SDPO exact full-vocab KL primitive.

Runs under torchrun; ``--nproc_per_node=1`` is a cheap TP=1 math check (value +
gradient + chunk equivalence), ``--nproc_per_node=2`` additionally exercises the
tensor-parallel vocab sharding + all-reduce. For alpha in {0 (forward KL,
KL(teacher||student)), 1 (reverse KL, KL(student||teacher))} it checks:

  * per-token KL matches a single-process full-vocab oracle;
  * the student-logit gradient matches the oracle's autograd gradient (sharded);
  * chunked (chunk_size=1) == non-chunked value and gradient.

Usage:
  torchrun --standalone --nproc_per_node=2 tests/special_standalone/test_sdpo_full_kl_dist.py
"""

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.trainer.distillation.megatron.losses import (
    _VocabParallelFullKLDivergence,
    vocab_parallel_full_kl_with_chunking,
)
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


def _full_vocab_kl_oracle(student_logits, teacher_log_probs, alpha):
    """Single-process full-vocab per-token KL over the whole vocabulary."""
    log_p = F.log_softmax(student_logits, dim=-1)
    log_q = teacher_log_probs
    if alpha == 0.0:  # KL(teacher || student)
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)  # KL(student || teacher)


def _shard(t, rank, shard_size):
    """Take this rank's contiguous vocab slice of a (..., V) tensor."""
    return t[..., rank * shard_size : (rank + 1) * shard_size].contiguous()


def main():
    local_rank, rank, world_size = initialize_global_process_group()
    mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(0)

    B, S, V = 2, 5, 64 * world_size
    assert V % world_size == 0
    shard_size = V // world_size

    # Identical full tensors on every rank (broadcast from rank 0 to be safe under per-device RNG).
    full_student_logits = torch.randn(B, S, V, device=device) * 0.8
    teacher_full_logits = torch.randn(B, S, V, device=device) * 1.1
    dist.broadcast(full_student_logits, src=0)
    dist.broadcast(teacher_full_logits, src=0)
    teacher_full_logps = F.log_softmax(teacher_full_logits, dim=-1)

    for alpha in (0.0, 1.0):
        # ---- oracle value + gradient over the full vocabulary ----
        student_oracle = full_student_logits.clone().requires_grad_(True)
        oracle_kl = _full_vocab_kl_oracle(student_oracle, teacher_full_logps, alpha)
        oracle_kl.sum().backward()
        oracle_grad_shard = _shard(student_oracle.grad, rank, shard_size)

        # ---- primitive on this rank's vocab shard ----
        vp_student = _shard(full_student_logits, rank, shard_size).clone().requires_grad_(True)
        vp_teacher = _shard(teacher_full_logps, rank, shard_size)
        kl = _VocabParallelFullKLDivergence.apply(vp_student, vp_teacher, alpha)
        assert kl.shape == (B, S), kl.shape
        torch.testing.assert_close(kl, oracle_kl.detach(), rtol=1e-4, atol=1e-4)

        kl.sum().backward()
        torch.testing.assert_close(vp_student.grad, oracle_grad_shard.detach(), rtol=1e-4, atol=1e-4)

        # ---- chunked (chunk_size=1) == non-chunked, value and gradient ----
        s_chunk = _shard(full_student_logits, rank, shard_size).clone().requires_grad_(True)
        s_whole = _shard(full_student_logits, rank, shard_size).clone().requires_grad_(True)
        kl_chunk = vocab_parallel_full_kl_with_chunking(s_chunk, vp_teacher, alpha, chunk_size=1)
        kl_whole = vocab_parallel_full_kl_with_chunking(s_whole, vp_teacher, alpha, chunk_size=S)
        torch.testing.assert_close(kl_chunk, kl_whole, rtol=1e-5, atol=1e-5)
        kl_chunk.sum().backward()
        kl_whole.sum().backward()
        torch.testing.assert_close(s_chunk.grad, s_whole.grad, rtol=1e-5, atol=1e-5)
        # chunked wrapper matches the oracle too
        torch.testing.assert_close(kl_chunk, oracle_kl.detach(), rtol=1e-4, atol=1e-4)

        if rank == 0:
            direction = "forward KL(teacher||student)" if alpha == 0.0 else "reverse KL(student||teacher)"
            print(f"[OK] alpha={alpha} ({direction}): value+grad match oracle; chunked==non-chunked (TP={world_size})")

    destroy_global_process_group()


if __name__ == "__main__":
    main()
