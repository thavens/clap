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

"""SDPO exact full-vocab KL primitive numerics at TP=1 on CPU.

Validates ``_VocabParallelFullKLDivergence`` / ``vocab_parallel_full_kl_with_chunking``
against a brute-force full-vocabulary oracle for both directions (alpha=0 forward KL
``KL(teacher||student)``; alpha=1 reverse KL ``KL(student||teacher)``): value vs oracle,
analytic gradient vs the oracle's autograd, chunked==non-chunked, non-negativity, and
self-divergence==0. At TP=1 the tensor-parallel all-reduce is a no-op and
``vocab_parallel_log_softmax`` reduces to a plain full-vocab log-softmax, so this exercises
the KL sum and the hand-written backward directly; the real TP sharding + all-reduce are
covered by ``tests/special_standalone/test_sdpo_full_kl_dist.py`` on GPU.

Internally the kernel computes in fp32, so tolerances are fp32-appropriate (~1e-4).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

mpu = pytest.importorskip("megatron.core.parallel_state")


@pytest.fixture(scope="module")
def tp1_cpu_group():
    """Single-process gloo group + megatron TP=1 so the vocab-parallel KL runs on CPU."""
    created_pg = False
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created_pg = True
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    mpu.destroy_model_parallel()
    if created_pg:
        dist.destroy_process_group()


def _full_vocab_kl_oracle(student_logits, teacher_log_probs, alpha):
    log_p = F.log_softmax(student_logits, dim=-1)
    log_q = teacher_log_probs
    if alpha == 0.0:  # KL(teacher || student)
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)  # KL(student || teacher)


def _inputs(seed=0, B=3, S=7, V=128):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(B, S, V, generator=g) * 0.8
    teacher = F.log_softmax(torch.randn(B, S, V, generator=g) * 1.1, dim=-1)
    return student, teacher


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_value_and_grad_match_full_vocab_oracle(tp1_cpu_group, alpha):
    from verl.trainer.distillation.megatron.losses import _VocabParallelFullKLDivergence

    student, teacher = _inputs()
    student_oracle = student.clone().requires_grad_(True)
    oracle = _full_vocab_kl_oracle(student_oracle, teacher, alpha)
    oracle.sum().backward()

    student_kernel = student.clone().requires_grad_(True)
    kl = _VocabParallelFullKLDivergence.apply(student_kernel, teacher, alpha)
    assert kl.shape == student.shape[:2]
    torch.testing.assert_close(kl, oracle.detach(), rtol=1e-4, atol=1e-4)

    kl.sum().backward()
    torch.testing.assert_close(student_kernel.grad, student_oracle.grad, rtol=1e-4, atol=1e-4)

    # A proper KL is non-negative.
    assert kl.min().item() >= -1e-5


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_chunked_equals_non_chunked(tp1_cpu_group, alpha):
    from verl.trainer.distillation.megatron.losses import vocab_parallel_full_kl_with_chunking

    student, teacher = _inputs(seed=2)
    seq_len = student.shape[1]
    s_chunk = student.clone().requires_grad_(True)
    s_whole = student.clone().requires_grad_(True)
    kl_chunk = vocab_parallel_full_kl_with_chunking(s_chunk, teacher, alpha, chunk_size=1)
    kl_whole = vocab_parallel_full_kl_with_chunking(s_whole, teacher, alpha, chunk_size=seq_len)
    torch.testing.assert_close(kl_chunk, kl_whole, rtol=1e-6, atol=1e-6)

    kl_chunk.sum().backward()
    kl_whole.sum().backward()
    torch.testing.assert_close(s_chunk.grad, s_whole.grad, rtol=1e-6, atol=1e-6)


def test_self_divergence_is_zero(tp1_cpu_group):
    """KL(p || p) == 0 for both directions when the teacher equals the student distribution."""
    from verl.trainer.distillation.megatron.losses import _VocabParallelFullKLDivergence

    student, _ = _inputs(seed=3)
    teacher = F.log_softmax(student, dim=-1)  # teacher distribution == student distribution
    for alpha in (0.0, 1.0):
        kl = _VocabParallelFullKLDivergence.apply(student.clone().requires_grad_(True), teacher, alpha)
        torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=1e-5)
