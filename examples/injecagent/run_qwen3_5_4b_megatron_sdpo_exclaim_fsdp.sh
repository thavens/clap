#!/usr/bin/env bash
# Exclamation-sentence-endings task | SDPO | Qwen3.5-4B | LoRA | Megatron-FSDP (ZeRO-3)
#
# Same free, deterministic SDPO exercise as run_qwen3_5_4b_megatron_sdpo_exclaim.sh, but the
# trainer runs Megatron-FSDP (ZeRO-3, data_parallel_sharding_strategy=optim_grads_params, the
# default when use_megatron_fsdp=true) over a pure data-parallel group instead of tensor
# parallelism. On the 2-GPU box this is TP=1 / DP=2: cross-GPU traffic is one gradient reduce per
# step (no per-layer TP all-reduce), and params / grads / optimizer state are sharded across the
# DP ranks. Megatron-FSDP also shards the frozen LoRA base weights (requires_grad=False params
# get a sharded model-weight buffer, just no grad/optimizer buffer).
#
# Getting the FSDP weight sync to vLLM working needed four verl-side pieces (all landed in the
# engine, no flags here): export_adapter_weights(cpu=False) so the bridge's fused-adapter split
# all-gathers over NCCL rather than gloo; a DTensor->full unshard at the sync boundary; a
# Megatron-Bridge build_adapter_conversion_tasks patch so its split/merge sees full tensors; and
# start_param_sync(force_sync=True) before export so ZeRO-3's sharded model-weight buffer is
# populated from the optimizer master (otherwise the local shards read as size-0 storage).
#
# Numerics check: rollout vs training logprob agreement (training/rollout_actor_probs_pearson_corr
# ~1.0, rollout_probs_diff_mean tiny) confirms the FSDP weight sync is bit-faithful; exclaim_fraction
# and score should climb exactly as in the TP baseline.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# TP=1 makes the two GPUs a data-parallel group; use_megatron_fsdp=true turns on ZeRO-3 sharding.
# Offloads stay off (params must be resident on GPU for the FSDP all-gather during weight sync).
# gpu_memory_utilization is trimmed to 0.6 to leave room for the sharded FSDP buffers alongside the
# woken vLLM engine; kl_chunk_size=128 and recompute-off match the tuned throughput config, and
# ppo/log-prob micro-batch of 8 keeps the full-vocab (unsharded at TP=1) SDPO KL logits in range.
exec "${SCRIPT_DIR}/run_qwen3_5_4b_megatron_sdpo_exclaim.sh" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=true \
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_param_gather=true \
    actor_rollout_ref.actor.megatron.param_offload=false \
    actor_rollout_ref.actor.megatron.optimizer_offload=false \
    actor_rollout_ref.actor.megatron.grad_offload=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    sdpo.kl_chunk_size=128 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=null \
    trainer.experiment_name=exclaim_sdpo_qwen35_4b_fsdp \
    "$@"
