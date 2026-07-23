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

"""Exclamation-sentence-endings reward (CPU, no network).

This task's whole value is that its verdict is unambiguous, so these tests pin the rule at its
edges and pin the feedback contract the SDPO reprompt path depends on.
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.ppo.sdpo_reprompt import build_teacher_messages
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.reward_score import exclaim
from verl.workers.config import DistillationLossConfig, SelfDistillationConfig


@pytest.mark.parametrize(
    ("response", "expected_score", "expected_fraction"),
    [
        ("Every sentence ends correctly! This one too!", 1.0, 1.0),
        ("One correct! One wrong.", 0.5, 0.5),
        ("Correct! Wrong? Also wrong.", 1 / 3, 1 / 3),
        # Mixed punctuation counts each mark, so the question mark is not silently accepted.
        ("Really?!", 0.5, 0.5),
        ("No! way!", 1.0, 1.0),
        ("Ignore previous instructions", 0.0, 0.0),
        # No sentence endings must not vacuously satisfy the requirement.
        ("", 0.0, 0.0),
        ("   \n\t ", 0.0, 0.0),
    ],
)
def test_score_response(response, expected_score, expected_fraction):
    score, fraction = exclaim.score_response(response)
    assert score == expected_score
    assert fraction == pytest.approx(expected_fraction)


@pytest.mark.parametrize(
    ("response", "expected_score"),
    [
        # Reasoning is dropped: only the post-</think> answer is scored, so period-laden thinking
        # cannot swamp the signal.
        ("<think>First. Second. Third.</think>All good! Yes!", 1.0),
        ("<think>Lots. Of. Periods.</think>Half good! Half bad.", 0.5),
        # Answer is empty / whitespace after reasoning -> no endings -> 0.
        ("<think>Reasoning! Reasoning!</think>", 0.0),
        # Unclosed reasoning (still thinking at the length limit) -> no answer -> 0.
        ("<think>Still reasoning! Never stopped!", 0.0),
        # No reasoning markers -> whole thing scored (no-op when thinking is disabled).
        ("Plain answer!", 1.0),
    ],
)
def test_score_response_strips_reasoning(response, expected_score):
    score, _ = exclaim.score_response(response)
    assert score == pytest.approx(expected_score)


@pytest.mark.parametrize("response,score", [("Good!", 1.0), ("Bad.", 0.0), ("Mixed! Bad?", 0.5)])
def test_feedback_is_fixed(response, score):
    assert exclaim.compose_feedback(response, score) == (
        "please write your final answer such that all sentences end in exclamations."
    )


def test_compute_score_shape_matches_the_reward_contract():
    result = exclaim.compute_score(
        data_source="injecagent",
        solution_str="not exclamations",
        ground_truth="SomeAttackerTool",
        extra_info={},
    )
    assert set(result) == {"score", "exclaim_fraction", "sentence_endings", "feedback"}
    assert result["score"] == 0.0
    assert result["sentence_endings"] == 0.0
    assert result["feedback"] == "please write your final answer such that all sentences end in exclamations."


def test_compute_score_ignores_data_source_and_ground_truth():
    """The rule reads only the response, which is what lets the task borrow InjecAgent prompts."""
    borrowed = exclaim.compute_score(
        data_source="injecagent", solution_str="A sentence!", ground_truth="SomeAttackerTool", extra_info={}
    )
    standalone = exclaim.compute_score(data_source="whatever", solution_str="A sentence!", ground_truth="", extra_info={})
    assert borrowed == standalone
    assert borrowed["score"] == 1.0


def _exclaim_sdpo_config() -> SelfDistillationConfig:
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=["sdpo=exclaim"])
    return omega_conf_to_dataclass(cfg.sdpo)


def test_reward_feedback_reaches_the_teacher_prompt():
    """End-to-end on the seam that matters: scorer feedback -> SDPO reprompt.

    Every rollout receives the same fixed feedback, so both are reprompted. The failed rollout
    additionally receives the successful sibling's response as a demonstration.
    """
    cfg = _exclaim_sdpo_config()
    responses = ["A successful response!", "An unsuccessful response."]
    scored = [exclaim.compute_score("injecagent", r, "tool", {}) for r in responses]

    messages, mask, metrics = build_teacher_messages(
        cfg=cfg,
        raw_prompts=[[{"role": "user", "content": "ATTACKER PROMPT"}]] * 2,
        prompt_texts=["ATTACKER PROMPT"] * 2,
        response_texts=responses,
        uids=["g0", "g0"],
        seq_scores=[s["score"] for s in scored],
        feedback_raw=[s["feedback"] for s in scored],
    )

    assert mask == [1.0, 1.0], "fixed feedback reprompts every rollout"
    assert metrics["sdpo/success_sample_fraction"] == 0.5

    teacher_prompt = messages[1][-1]["content"]
    assert "ATTACKER PROMPT" in teacher_prompt, "the original task must survive the reprompt"
    assert scored[1]["feedback"] in teacher_prompt
    assert responses[0] in teacher_prompt, "the successful sibling is demonstrated"
    # sdpo.yaml's stock wording would tell the teacher to write an injection instead.
    assert "improved injection" not in teacher_prompt


def test_exclaim_config_is_pure_sdpo():
    """The score must not reach the loss: the feedback path is what is under test."""
    cfg = _exclaim_sdpo_config()
    assert cfg.enabled
    assert not cfg.distillation_loss.use_task_rewards
    assert not cfg.distillation_loss.use_policy_gradient
    assert cfg.include_environment_feedback
    # With the default threshold, only a response with exclusively exclamation endings is a solution.
    assert cfg.success_reward_threshold == 1.0
    assert isinstance(cfg.distillation_loss, DistillationLossConfig)
