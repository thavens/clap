#!/usr/bin/env bash
# InjecAgent CLAP | Dr GRPO | Qwen3.5-4B-Base | LoRA | Megatron Bridge

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B-Base}
TRAIN_PATH=${TRAIN_PATH:-$HOME/data/injecagent/train.parquet}
VAL_PATH=${VAL_PATH:-$HOME/data/injecagent/eval.parquet}

DATA=(
    data.train_files="${TRAIN_PATH}"
    data.val_files="${VAL_PATH}"
    data.train_batch_size=16
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=False
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.lora.rank=64
    actor_rollout_ref.model.lora.alpha=32
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.lora.target_modules='["*language_model.*.linear_qkv","*language_model.*.linear_proj","*language_model.*.linear_fc1","*language_model.*.linear_fc2"]'
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=5e-5
    actor_rollout_ref.actor.optim.weight_decay=0
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024
    actor_rollout_ref.actor.freeze_vision_tower=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.loss_scale_factor=1024
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.top_p=1
    actor_rollout_ref.rollout.max_model_len=1024
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,32,64]'
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_cutedsl_warmup=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
)

REWARD=(
    reward.num_workers=1
    reward.custom_reward_function.path="${REPO_ROOT}/verl/utils/reward_score/injecagent.py"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='[console,wandb]'
    trainer.project_name=clap
    trainer.experiment_name=injecagent_gpt5nano_clap
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.val_before_train=True
    trainer.test_freq=32
    trainer.save_freq=32
    trainer.total_epochs=1000
    trainer.total_training_steps=152
    trainer.resume_mode=disable
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    model_engine=megatron \
    "$@"
