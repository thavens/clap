## motivation

We want to move from SkyRL to verl because it's much harder to do on policy self distillation RL methods.

## Task

For intial testing of this library please migrate the injecagent environment from the one we have in @..SkyRL .
Basically everything should be hard coded for this first attempt.
use the script from ../rl-hammer-hardening/scripts/injecagent/train/gpt5nano_clap.py as the behaviour copy that we are trying to implement.
Make sure that we use rank 64 LoRA, same steps, and data.
Use the megatron / megatron-bridge trainer backend.

## Test

Smoke test by running an iteration over RunPod.
use the wandb project clap.
