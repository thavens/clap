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

"""Real multi-GPU validation of vocab_parallel_topk_log_probs (the SDPO self-teacher top-k).

Splits a full logits tensor across ``world_size`` ranks (simulating vocab sharding) using a
real NCCL process group -- NOT megatron -- and checks the distributed all_gather + two-stage
top-k + TP-reduced log-softmax matches the single-process global top-k.

Launch (2 GPUs):
    torchrun --standalone --nproc_per_node=2 tests/special_standalone/test_sdpo_vocab_topk_dist.py
"""

import os

import torch
import torch.distributed as dist

from verl.utils.megatron.tensor_parallel import vocab_parallel_topk_log_probs


def test_sdpo_vocab_topk_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    b, t, vocab, k = 4, 16, 4096, 100
    assert vocab % world_size == 0
    shard = vocab // world_size

    # Every rank builds the SAME full logits (seeded), then keeps only its vocab shard.
    gen = torch.Generator().manual_seed(2026)
    full_logits = torch.randn(b, t, vocab, generator=gen, dtype=torch.float32)
    full_logits += torch.linspace(0, 1e-3, vocab)  # break ties
    full_logits = full_logits.to(device)

    vocab_start = rank * shard
    vp_logits = full_logits[..., vocab_start : vocab_start + shard].contiguous()

    topk_logps, topk_ids = vocab_parallel_topk_log_probs(vp_logits, k, group=dist.group.WORLD)

    # Single-process reference on the full tensor.
    ref_logps, ref_ids = torch.topk(torch.log_softmax(full_logits, dim=-1), k=k, dim=-1)

    torch.testing.assert_close(topk_logps, ref_logps, rtol=1e-4, atol=1e-4)
    assert torch.equal(topk_ids, ref_ids), "top-k ids diverged from global reference"

    # Every rank must agree (the function returns the same global top-k on all ranks).
    gathered = [torch.empty_like(topk_ids) for _ in range(world_size)]
    dist.all_gather(gathered, topk_ids)
    for other in gathered:
        assert torch.equal(other, topk_ids), "ranks returned different top-k ids"

    if rank == 0:
        print(
            f"[OK] vocab_parallel_topk_log_probs matches global top-k "
            f"(world_size={world_size}, vocab={vocab}, k={k}, shape={tuple(topk_logps.shape)})"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_sdpo_vocab_topk_dist()
