#!/usr/bin/env bash
# Exclamation-sentence-endings task | SDPO | Qwen3.5-4B (instruct, reasoning off) | LoRA | Megatron
#
# A free, deterministic end-to-end exercise of the SDPO loop. The policy keeps the InjecAgent
# CLAP attacker prompts, but the reward is swapped for a local graded predicate: the fraction of
# `.`, `?`, and `!` sentence-ending marks that are `!` (verl/utils/reward_score/exclaim.py). No
# hosted defender, so no API key and no per-step cost -- iterate on this as much as you like.
#
# The loss defers entirely to SDPO feedback. Pure SDPO (inherited from the SDPO launcher:
# use_task_rewards=false, use_policy_gradient=false) means the score never enters the loss; it
# only picks sibling demonstrations and fills the metrics. What moves the weights is the
# divergence to the self-teacher, which is this policy re-reading its own task with the scorer's
# per-rollout feedback appended.
#
# Pass condition: exclaim_fraction and score climb, and sampled sentences visibly end in `!`
# instead of `.` or `?`. sdpo/reprompt_sample_fraction should start near 1.0 (nearly every
# rollout fails at first, so nearly every rollout has feedback) and fall as the policy solves the task. If
# reprompt_sample_fraction is healthy but exclaim_fraction is flat over ~20 steps, the fault is
# in the objective or the top-k transport, not in the reprompt construction.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# The exclaim task only moves the policy if it can follow the scorer's in-context feedback. The
# 4B *base* model follows that feedback too weakly to be taught by pure-SDPO distillation, so this
# task uses the instruct model. Reasoning stays disabled via the base launcher's
# enable_thinking=False -- the exclaim reward scores only the post-</think> answer, so thinking
# would only flood it with period-ended sentences. Callers that export MODEL_PATH (e.g. the sky
# stages, which point at the pre-downloaded local dir) still win.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

# Delegates to the SDPO launcher (which delegates to the Dr-GRPO one), so model / LoRA /
# Megatron / rollout settings stay in one place and this file is only the task delta.
# `sdpo=exclaim` swaps the sdpo config group for this task's prompt wording; the SDPO launcher's
# `sdpo.*` value overrides still apply on top of it.
exec "${SCRIPT_DIR}/run_qwen3_5_4b_megatron_sdpo.sh" \
    sdpo=exclaim \
    reward.custom_reward_function.path="${REPO_ROOT}/verl/utils/reward_score/exclaim.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.experiment_name=exclaim_sdpo_qwen35_4b \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=20 \
    "$@"
