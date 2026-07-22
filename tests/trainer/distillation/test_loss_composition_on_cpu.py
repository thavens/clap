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

from types import SimpleNamespace

import pytest
import torch

import verl.trainer.distillation.losses as losses


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_task_and_distillation_losses_support_convex_interpolation(monkeypatch, alpha):
    task_loss = torch.tensor(4.0)
    distill_loss = torch.tensor(8.0)
    loss_config = SimpleNamespace(
        use_task_rewards=True,
        use_policy_gradient=False,
        task_loss_coef=1.0 - alpha,
        distillation_loss_coef=alpha,
    )
    config = SimpleNamespace(distillation_loss=loss_config)

    monkeypatch.setattr(losses, "ppo_loss", lambda *_args, **_kwargs: (task_loss, {}))
    monkeypatch.setattr(losses, "distillation_loss", lambda *_args, **_kwargs: (distill_loss, {}))

    result, metrics = losses.distillation_ppo_loss(None, config)

    torch.testing.assert_close(result, torch.tensor((1.0 - alpha) * 4.0 + alpha * 8.0))
    assert metrics["distillation/task_loss_coef"].aggregate() == pytest.approx(1.0 - alpha)
    assert metrics["distillation/loss_coef"].aggregate() == pytest.approx(alpha)
