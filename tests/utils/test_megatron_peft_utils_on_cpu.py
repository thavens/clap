# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch

from verl.utils.megatron_peft_utils import add_base_layer_suffix, convert_megatron_to_hf_target_modules


def test_convert_wildcard_qualified_megatron_lora_targets():
    targets = [
        "*language_model.*.linear_qkv",
        "*language_model.*.linear_proj",
        "*language_model.*.linear_fc1",
        "*language_model.*.linear_fc2",
    ]

    assert convert_megatron_to_hf_target_modules(targets) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_add_base_layer_suffix_for_qwen35_linear_attention_weights():
    weights = [
        (f"model.layers.0.linear_attn.{name}.weight", torch.empty(1))
        for name in ("conv1d", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    ]

    names = [name for name, _ in add_base_layer_suffix(iter(weights), "qwen3_5")]

    assert names == [name.replace(".weight", ".base_layer.weight") for name, _ in weights]
