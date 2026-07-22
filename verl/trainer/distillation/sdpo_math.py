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

"""Pure divergence math for SDPO self-distillation.

Backend-agnostic tensor functions, no distributed / Megatron / DataProto deps,
so they are unit-testable on CPU against the reference implementation in
``thavens/SDPO`` (``verl/trainer/ppo/core_algos.py::compute_self_distillation_loss``).

Two entry points:

- ``sdpo_topk_divergence``: the general SDPO per-token divergence between a
  student and a (stop-gradient) teacher whose top-k log-probs are given on a
  *shared* id set. Interpolates forward KL (alpha=0), reverse KL (alpha=1) and
  generalized Jensen-Shannon (in between), with an optional tail bucket. This is
  the *reference* formulation: the training path does not call it (see below), it
  exists as the oracle that ``forward_kl_from_masses`` is tested against.

- ``forward_kl_from_masses``: for the alpha=0 case only, reconstruct SDPO's
  tail-bucketed / renormalized forward KL from the *truncated* forward KL and
  the student/teacher top-k masses that verl's existing vocab-parallel kernel
  (``_VocabParallelKLDivergence``) already returns. This lets the Megatron path
  reuse that proven kernel verbatim and apply the SDPO correction as a cheap,
  non-distributed post-processing step. Algebra:

    renormalized:  KL = trunc_fkl / Zq - log Zq + log Zp
    tail-bucketed: KL = trunc_fkl + (1 - Zq) * (log(1 - Zq) - log(1 - Zp))

  where ``trunc_fkl = sum_{i in teacher-topk} q_i (log q_i - log p_i)`` (teacher
  probs un-renormalized), ``Zq = sum q_i`` (teacher_mass) and
  ``Zp = sum p_i`` (student_mass, the student probability that lands on the
  teacher's top-k ids).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "add_tail_bucket",
    "renorm_topk_log_probs",
    "sdpo_topk_divergence",
    "forward_kl_from_masses",
]

# Guards log(0) when a top-k set already holds ~all the probability mass.
_TAIL_LOG_MAX = -1e-7


def add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail log-prob ``log(1 - sum(exp(log_probs)))`` along the last dim.

    Uses ``logsumexp`` + ``expm1`` for numerical stability, mirroring the SDPO
    reference. The result is a proper distribution (sums to 1) over top-k + tail.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=_TAIL_LOG_MAX)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalize top-k log-probs so they sum to 1 over the retained candidates."""
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def sdpo_topk_divergence(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    alpha: float,
    add_tail: bool,
) -> torch.Tensor:
    """Per-token SDPO divergence between student and (detached) teacher top-k dists.

    The training path does not call this: the Megatron kernel returns summary tensors from
    which ``forward_kl_from_masses`` reconstructs the alpha=0 case more cheaply. This is the
    direct formulation kept as the reference oracle both for that differential test and for
    the alpha>0 variants, which are not yet wired into the top-k loss path.

    Args:
        student_topk_log_probs: (..., K) student log-probs on a shared id set.
            Carries gradient (it is the model output).
        teacher_topk_log_probs: (..., K) teacher log-probs on the *same* id set.
            Expected to be detached by the caller (stop-gradient teacher).
        alpha: 0.0 -> forward KL ``KL(teacher||student)``; 1.0 -> reverse KL
            ``KL(student||teacher)``; in between -> generalized Jensen-Shannon.
        add_tail: append a tail bucket (proper distribution) vs. renormalize top-k.

    Returns:
        (...,) per-token divergence.
    """
    if add_tail:
        student = add_tail_bucket(student_topk_log_probs)
        teacher = add_tail_bucket(teacher_topk_log_probs)
    else:
        student = renorm_topk_log_probs(student_topk_log_probs)
        teacher = renorm_topk_log_probs(teacher_topk_log_probs)

    if alpha == 0.0:
        # KL(teacher || student); gradient flows through student (the `input` arg).
        kl = F.kl_div(student, teacher, reduction="none", log_target=True)
    elif alpha == 1.0:
        # KL(student || teacher); gradient flows through student (the `target` arg,
        # which requires grad and appears as exp(student)*(student - teacher)).
        kl = F.kl_div(teacher, student, reduction="none", log_target=True)
    else:
        a = torch.tensor(alpha, dtype=student.dtype, device=student.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student + torch.log(1 - a), teacher + torch.log(a)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, a)

    return kl.sum(-1)


def forward_kl_from_masses(
    truncated_forward_kl: torch.Tensor,
    student_mass: torch.Tensor,
    teacher_mass: torch.Tensor,
    add_tail: bool,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Reconstruct SDPO's forward KL (alpha=0) from truncated FKL + top-k masses.

    See module docstring for the algebra. Equivalent to
    ``sdpo_topk_divergence(alpha=0)`` when the shared id set is the teacher's
    top-k, but computed from the summary tensors verl's Megatron kernel already
    returns (so no student log-probs at teacher ids are needed downstream).

    Args:
        truncated_forward_kl: (...,) ``sum_i q_i (log q_i - log p_i)`` over the
            teacher top-k (teacher probs NOT renormalized).
        student_mass: (...,) ``Zp = sum_i p_i`` (student mass on teacher top-k ids).
        teacher_mass: (...,) ``Zq = sum_i q_i``.
        add_tail: tail bucket vs. renormalize.
    """
    if add_tail:
        q_tail = (1.0 - teacher_mass).clamp_min(eps)
        p_tail = (1.0 - student_mass).clamp_min(eps)
        return truncated_forward_kl + q_tail * (q_tail.log() - p_tail.log())
    zq = teacher_mass.clamp_min(eps)
    zp = student_mass.clamp_min(eps)
    return truncated_forward_kl / zq - zq.log() + zp.log()
