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

"""Exclamation-sentence-endings task: a local, deterministic environment for exercising SDPO.

The policy is asked (via whatever prompt the dataset carries -- in practice the InjecAgent CLAP
attacker prompt, reused unchanged) to end sentences with ``!`` rather than ``.`` or ``?``. The
reward is the fraction of those three punctuation marks that are ``!``: no hosted model, no
network, no API spend, no nondeterminism.

Why this exists: on the real CLAP task, "is SDPO working?" can only be inferred from a loss
curve, because the frozen defender makes the target distribution unknown and every step costs
money. Here the target is a property you can read straight off a rollout, and a full run is
free. It isolates the plumbing under test -- reprompt construction, teacher-forced scoring,
top-k transport, the divergence -- from the reward's own difficulty.

The training signal is meant to come entirely from ``feedback``: run this with pure SDPO
(``use_task_rewards=false``, ``use_policy_gradient=false``), where ``score`` reaches the loss
through nothing but the self-teacher's feedback-conditioned distribution. ``score`` still drives
sibling-solution selection and the logged metrics.

Responses without any recognized sentence-ending punctuation score 0.0 rather than satisfying
the requirement vacuously.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["compose_feedback", "compute_score", "score_response"]

_FEEDBACK = "please write your final answer such that all sentences end in exclamations."
_SENTENCE_ENDINGS = frozenset(".!?")


def answer_after_reasoning(solution_str: str) -> str:
    """Return the final answer, dropping any ``<think>...</think>`` reasoning span.

    With reasoning enabled the response is ``<think>{reasoning}</think>{answer}``; the reasoning
    is full of ordinary period-ending sentences that would swamp the exclamation signal (and never
    let a rollout clear the demonstration threshold). Score only the post-``</think>`` answer.
    An unclosed ``<think>`` (the model was still reasoning at the length limit) means no answer was
    produced -> empty string -> score 0.0. A response with no reasoning markers is scored whole, so
    this is a no-op when reasoning is disabled.
    """
    if "</think>" in solution_str:
        return solution_str.split("</think>", 1)[1]
    if "<think>" in solution_str:
        return ""
    return solution_str


def score_response(solution_str: str) -> tuple[float, float]:
    """Return ``(score, exclaim_fraction)`` for one response.

    Each ``.``, ``?``, or ``!`` is treated as a sentence-ending mark. The reward is the fraction
    of those marks that are ``!``. Counting marks directly makes mixed punctuation such as ``?!``
    receive partial credit rather than silently treating the question mark as acceptable. Only the
    final answer is scored (any ``<think>...</think>`` reasoning span is dropped first).
    """
    answer = answer_after_reasoning(solution_str)
    endings = [char for char in answer if char in _SENTENCE_ENDINGS]
    if not endings:
        return 0.0, 0.0
    fraction = sum(char == "!" for char in endings) / len(endings)
    return fraction, fraction


def compose_feedback(solution_str: str, score: float) -> str:
    """Return the fixed feedback used by the SDPO self-teacher prompt."""
    del solution_str, score
    return _FEEDBACK


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any],
    **_: Any,
) -> dict[str, float | str]:
    """Score one rollout by the fraction of sentence-ending marks that are exclamations.

    ``data_source`` and ``ground_truth`` are accepted to satisfy verl's reward contract and
    deliberately ignored: the rule reads only the response text, so it is well-defined on any
    dataset and needs no preprocessing pass of its own. That is what lets this task borrow the
    InjecAgent prompts (``data_source='injecagent'``) as-is.
    """
    score, fraction = score_response(solution_str)
    answer = answer_after_reasoning(solution_str)
    return {
        "score": score,
        "exclaim_fraction": fraction,
        "sentence_endings": float(sum(char in _SENTENCE_ENDINGS for char in answer)),
        # SDPO feedback rides through reward_extra_info into non_tensor_batch, where the trainer
        # picks it up to build the self-teacher's feedback-augmented prompt.
        "feedback": compose_feedback(solution_str, score),
    }
