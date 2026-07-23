#!/usr/bin/env bash
# InjecAgent CLAP | SDPO (alpha=0 forward-KL self-distillation) | Qwen3.5-4B-Base | LoRA | Megatron
#
# Pure SDPO: the attacker policy is distilled toward its own feedback-conditioned distribution
# (the frozen defender's reply is the feedback), with no task-reward policy-gradient term. Flip
# sdpo.distillation_loss.use_task_rewards=true (+ task/distillation coefficients) to compose with GRPO.
#
# Delegates to the base Dr-GRPO launch script and only overrides the SDPO knobs, so model /
# LoRA / Megatron / rollout / reward settings stay in one place.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Sequence parallelism must be OFF for the SDPO full-vocab KL. The loss runs inside the megatron
# logit processor on `student_logits`; with sequence_parallel=True (the TP>1 default) those logits
# are scattered to seqlen/TP per rank, while the co-located teacher's log-softmax is remapped at
# full sequence length, so the two can't be aligned without replicating Megatron's internal SP
# scatter. Disabling SP makes the LM-head logits full-length, matching the teacher. Pipeline and
# context parallel must also be 1 (the co-located teacher forward asserts this). (Small
# activation-memory cost at TP=2; correctness > perf for this recipe.)
exec "${SCRIPT_DIR}/run_qwen3_5_4b_megatron.sh" \
    sdpo.enabled=true \
    sdpo.alpha=0.0 \
    sdpo.max_reprompt_len=2048 \
    sdpo.include_environment_feedback=true \
    sdpo.distillation_loss.use_task_rewards=false \
    sdpo.distillation_loss.use_policy_gradient=false \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    trainer.experiment_name=injecagent_sdpo_qwen35_4b \
    "$@"
