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

"""CPU numerics tests for SDPO self-distillation divergence math.

Locks ``verl.trainer.distillation.sdpo_math`` against brute-force divergences computed
in probability space, and locks the mass-based reconstruction that the Megatron path
actually runs against the direct top-k formulation.
"""

import pytest
import torch

from verl.trainer.distillation.sdpo_math import (
    add_tail_bucket,
    forward_kl_from_masses,
    renorm_topk_log_probs,
    sdpo_topk_divergence,
)


def _brute_force_divergence(student_topk_logp, teacher_topk_logp, alpha, add_tail):
    """Ground truth for ``sdpo_topk_divergence``, derived from the definitions.

    Deliberately shares no machinery with the implementation: it builds explicit
    probability vectors and sums ``a * log(a/b)`` elementwise, instead of going through
    ``logsumexp`` / ``expm1`` / ``F.kl_div`` / ``torch.lerp``. A bug in the implementation's
    log-space algebra therefore cannot be mirrored here.
    """
    p = student_topk_logp.double().exp()
    q = teacher_topk_logp.double().exp()
    if add_tail:
        p = torch.cat([p, 1.0 - p.sum(-1, keepdim=True)], dim=-1)
        q = torch.cat([q, 1.0 - q.sum(-1, keepdim=True)], dim=-1)
    else:
        p = p / p.sum(-1, keepdim=True)
        q = q / q.sum(-1, keepdim=True)

    def kl(a, b):
        return (a * (a.log() - b.log())).sum(-1)

    if alpha == 0.0:
        return kl(q, p)  # forward KL: KL(teacher || student)
    if alpha == 1.0:
        return kl(p, q)  # reverse KL: KL(student || teacher)
    mixture = (1.0 - alpha) * p + alpha * q
    return (1.0 - alpha) * kl(p, mixture) + alpha * kl(q, mixture)


def _random_topk(seed, shape=(4, 7), vocab=53, k=11):
    """Return (student_at_teacher_ids, teacher_topk_logp, full student/teacher logp, teacher ids)."""
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(*shape, vocab, generator=g, dtype=torch.float64)
    teacher_logits = torch.randn(*shape, vocab, generator=g, dtype=torch.float64)
    student_logp = torch.log_softmax(student_logits, dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
    teacher_topk_logp, teacher_ids = torch.topk(teacher_logp, k=k, dim=-1)
    student_at_teacher = torch.gather(student_logp, -1, teacher_ids)
    return student_at_teacher, teacher_topk_logp, student_logp, teacher_logp, teacher_ids


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("add_tail", [True, False])
def test_matches_brute_force_divergence(alpha, add_tail):
    student_at_teacher, teacher_topk_logp, *_ = _random_topk(seed=1 if alpha == 0.0 else 2)
    got = sdpo_topk_divergence(student_at_teacher, teacher_topk_logp, alpha=alpha, add_tail=add_tail)
    ref = _brute_force_divergence(student_at_teacher, teacher_topk_logp, alpha=alpha, add_tail=add_tail)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


def test_add_tail_bucket_conserves_mass():
    g = torch.Generator().manual_seed(7)
    logp = torch.log_softmax(torch.randn(3, 5, 40, generator=g, dtype=torch.float64), dim=-1)
    topk_logp, _ = torch.topk(logp, k=8, dim=-1)
    with_tail = add_tail_bucket(topk_logp)
    mass = with_tail.exp().sum(-1)
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-9, atol=1e-9)


def test_renorm_sums_to_one():
    g = torch.Generator().manual_seed(8)
    logp = torch.log_softmax(torch.randn(2, 4, 30, generator=g, dtype=torch.float64), dim=-1)
    topk_logp, _ = torch.topk(logp, k=6, dim=-1)
    mass = renorm_topk_log_probs(topk_logp).exp().sum(-1)
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("add_tail", [True, False])
def test_forward_kl_from_masses_matches_topk_divergence(add_tail):
    """The mass-based reconstruction (Megatron reuse path) must equal the direct top-k divergence."""
    student_at_teacher, teacher_topk_logp, *_ = _random_topk(seed=13)

    # Summary tensors that verl's vocab-parallel kernel returns (no clamp).
    teacher_topk_probs = teacher_topk_logp.exp()
    student_topk_probs = student_at_teacher.exp()
    truncated_fkl = (teacher_topk_probs * (teacher_topk_logp - student_at_teacher)).sum(-1)
    teacher_mass = teacher_topk_probs.sum(-1)
    student_mass = student_topk_probs.sum(-1)

    from_masses = forward_kl_from_masses(truncated_fkl, student_mass, teacher_mass, add_tail=add_tail)
    direct = sdpo_topk_divergence(student_at_teacher, teacher_topk_logp, alpha=0.0, add_tail=add_tail)
    torch.testing.assert_close(from_masses, direct, rtol=1e-8, atol=1e-8)


def test_alpha0_full_vocab_equals_true_forward_kl():
    """With K=V and no tail (renorm is identity), alpha=0 equals the exact forward KL."""
    student_at_teacher, teacher_topk_logp, student_logp, teacher_logp, _ = _random_topk(seed=21, vocab=17, k=17)
    got = sdpo_topk_divergence(student_at_teacher, teacher_topk_logp, alpha=0.0, add_tail=False)
    true_fkl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    torch.testing.assert_close(got, true_fkl, rtol=1e-8, atol=1e-8)


def test_zero_divergence_when_teacher_equals_student():
    """Non-reprompted samples have teacher == student -> zero loss (mask is a no-op there)."""
    g = torch.Generator().manual_seed(5)
    logp = torch.log_softmax(torch.randn(3, 6, 40, generator=g, dtype=torch.float64), dim=-1)
    topk_logp, _ = torch.topk(logp, k=9, dim=-1)
    for alpha in (0.0, 0.5, 1.0):
        div = sdpo_topk_divergence(topk_logp, topk_logp, alpha=alpha, add_tail=True)
        torch.testing.assert_close(div, torch.zeros_like(div), rtol=0, atol=1e-9)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_gradient_reaches_student_with_detached_teacher(alpha):
    """The caller supplies a stop-gradient teacher; the student must still get a real gradient.

    Notably for alpha=1.0, where the student is passed as ``F.kl_div``'s ``target``.
    """
    student_at_teacher, teacher_topk_logp, *_ = _random_topk(seed=33)
    student = student_at_teacher.clone().requires_grad_(True)
    teacher = teacher_topk_logp.detach()
    div = sdpo_topk_divergence(student, teacher, alpha=alpha, add_tail=True)
    div.sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
