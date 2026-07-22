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

"""Reprompt construction for SDPO self-distillation.

Framework-agnostic helpers (plain lists, no torch / DataProto / tokenizer) so
the same logic serves both the v1 (TransferQueue) and v0 trainers. The trainer
supplies decoded response texts, prompt texts, per-sample uids, per-sample
scalar rewards, and the per-sample feedback strings; these helpers decide which
samples get reprompted and assemble the chat messages. Tokenization and the
concat-with-responses step stay in the trainer, which owns the tokenizer.

Ported from ``thavens/SDPO`` ``verl/trainer/ppo/ray_trainer.py`` (the
``_collect_feedback`` / ``_collect_solutions_by_uid`` / ``_get_solution`` /
``_maybe_build_self_distillation_batch`` group).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from verl.workers.config import SelfDistillationConfig

__all__ = ["build_teacher_messages"]

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def remove_thinking_trace(text: str) -> str:
    """Remove ``<think>...</think>`` spans (and trailing whitespace) from text."""
    return _THINK_RE.sub("", text)


def collect_feedback(
    include_environment_feedback: bool,
    feedback_raw: Optional[list[Any]],
    batch_size: int,
) -> list[Optional[str]]:
    """Normalize per-sample environment feedback into a list of non-empty strings or None."""
    feedback_list: list[Optional[str]] = [None] * batch_size
    if include_environment_feedback and feedback_raw is not None:
        assert len(feedback_raw) == batch_size, (
            f"feedback is misaligned with the batch: got {len(feedback_raw)} entries for {batch_size} samples"
        )
        for i, value in enumerate(feedback_raw):
            if value and isinstance(value, str) and value.strip():
                feedback_list[i] = value
    return feedback_list


def collect_solutions_by_uid(
    uids: list[Any],
    seq_scores: list[float],
    success_reward_threshold: float,
) -> dict[Any, list[int]]:
    """Bucket sample indices by uid, keeping only rollouts whose scalar reward >= threshold."""
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def get_solution(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids: list[Any],
    response_texts: list[str],
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Pick a successful sibling response from the same uid group as a demonstration."""
    uid = uids[idx]
    solution_idxs = success_by_uid.get(uid, [])
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    # Taking the first successful demonstration effectively selects a random one,
    # since rollouts within a group are not ordered by quality.
    solution_str = response_texts[solution_idxs[0]]
    if remove_thinking_from_demonstration:
        solution_str = remove_thinking_trace(solution_str)
    return solution_str


def build_teacher_messages(
    cfg: SelfDistillationConfig,
    raw_prompts: list[list[dict]],
    prompt_texts: list[str],
    response_texts: list[str],
    uids: list[Any],
    seq_scores: list[float],
    feedback_raw: Optional[list[Any]],
) -> tuple[list[list[dict]], list[float], dict[str, float]]:
    """Assemble per-sample feedback-augmented teacher chat messages.

    Args:
        cfg: SDPO config.
        raw_prompts: per-sample original chat messages (``non_tensor_batch['raw_prompt']``);
            the final message is the attacker task and everything before it is preserved
            (e.g. system messages).
        prompt_texts: per-sample text of the final (user) prompt message.
        response_texts: per-sample decoded response strings.
        uids: per-sample group id.
        seq_scores: per-sample scalar reward (sequence sum).
        feedback_raw: per-sample environment feedback strings (or None), aligned to samples.

    Returns:
        (messages, self_distillation_mask, metrics) where ``self_distillation_mask[i]`` is
        1.0 iff sample ``i`` received a reprompt (has a solution or usable feedback), else 0.0.
        Non-reprompted samples keep their original prompt so teacher == student and the loss
        is zeroed by the mask.
    """
    batch_size = len(uids)

    feedback_list = collect_feedback(cfg.include_environment_feedback, feedback_raw, batch_size)
    success_by_uid = collect_solutions_by_uid(uids, seq_scores, cfg.success_reward_threshold)
    solution_strs = [
        get_solution(
            i,
            success_by_uid,
            uids,
            response_texts,
            cfg.dont_reprompt_on_self_success,
            cfg.remove_thinking_from_demonstration,
        )
        for i in range(batch_size)
    ]

    feedback_only_without_solution = cfg.environment_feedback_only_without_solution
    messages: list[list[dict]] = []
    feedback_used: list[bool] = []
    self_distillation_mask: list[float] = []
    for i in range(batch_size):
        system_messages = list(raw_prompts[i][:-1])
        has_solution = solution_strs[i] is not None
        has_feedback = feedback_list[i] is not None
        use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
        feedback_used.append(use_feedback)
        # A sample is reprompted (and so distilled) iff it has something to be reprompted with.
        reprompted = use_feedback or has_solution
        self_distillation_mask.append(1.0 if reprompted else 0.0)

        if reprompted:
            solution_section = (
                cfg.solution_template.format(successful_previous_attempt=solution_strs[i]) if has_solution else ""
            )
            feedback_section = cfg.feedback_template.format(feedback_raw=feedback_list[i]) if use_feedback else ""
            reprompt_text = cfg.reprompt_template.format(
                prompt=prompt_texts[i],
                solution=solution_section,
                feedback=feedback_section,
            )
        else:
            reprompt_text = prompt_texts[i]

        messages.append(system_messages + [{"role": "user", "content": reprompt_text}])

    unique_uids = set(uids)
    num_with_solution = sum(1 for s in solution_strs if s is not None)
    num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
    num_with_feedback_used = sum(1 for used in feedback_used if used)
    metrics = {
        "sdpo/success_group_fraction": (
            len([uid for uid in unique_uids if len(success_by_uid.get(uid, [])) > 0]) / len(unique_uids)
        ),
        "sdpo/success_sample_fraction": num_with_solution / batch_size,
        "sdpo/feedback_available_fraction": num_with_feedback_available / batch_size,
        "sdpo/feedback_used_fraction": num_with_feedback_used / batch_size,
        "sdpo/reprompt_sample_fraction": sum(self_distillation_mask) / batch_size,
    }
    return messages, self_distillation_mask, metrics
