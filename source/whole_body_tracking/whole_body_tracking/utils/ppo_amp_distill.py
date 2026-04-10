from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AmpDistillConfig:
    """Configuration for stage-aware AMP/distillation auxiliaries."""

    enable_amp: bool = False
    enable_distill: bool = False
    distill_action_coef: float = 1.0
    distill_feature_coef: float = 0.5
    amp_coef: float = 1.0


class AMPDiscriminator(nn.Module):
    """Small MLP discriminator for Adversarial Motion Prior.

    This module is framework-agnostic and can be plugged into PPO update steps.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def amp_discriminator_loss(
    discriminator: AMPDiscriminator,
    policy_features: torch.Tensor,
    reference_features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Binary discriminator loss for AMP.

    Args:
        discriminator: AMP discriminator network.
        policy_features: Features sampled from policy rollouts.
        reference_features: Features sampled from reference motions.

    Returns:
        total loss and a metrics dictionary.
    """

    logits_policy = discriminator(policy_features)
    logits_ref = discriminator(reference_features)

    labels_policy = torch.zeros_like(logits_policy)
    labels_ref = torch.ones_like(logits_ref)

    loss_policy = F.binary_cross_entropy_with_logits(logits_policy, labels_policy)
    loss_ref = F.binary_cross_entropy_with_logits(logits_ref, labels_ref)
    total = 0.5 * (loss_policy + loss_ref)

    with torch.no_grad():
        prob_policy = torch.sigmoid(logits_policy).mean()
        prob_ref = torch.sigmoid(logits_ref).mean()

    metrics = {
        "amp_loss": total.detach(),
        "amp_prob_policy": prob_policy.detach(),
        "amp_prob_ref": prob_ref.detach(),
    }
    return total, metrics


def action_distill_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
) -> torch.Tensor:
    """L2 action distillation loss."""
    return F.mse_loss(student_actions, teacher_actions)


def feature_distill_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """L2 intermediate feature distillation loss."""
    return F.mse_loss(student_features, teacher_features)


def combine_auxiliary_loss(
    cfg: AmpDistillConfig,
    amp_loss: torch.Tensor | None = None,
    action_loss: torch.Tensor | None = None,
    feature_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose auxiliary losses from enabled components."""

    device = None
    for t in (amp_loss, action_loss, feature_loss):
        if t is not None:
            device = t.device
            break
    total = torch.zeros((), device=device if device is not None else "cpu")

    if cfg.enable_amp and amp_loss is not None:
        total = total + cfg.amp_coef * amp_loss

    if cfg.enable_distill:
        if action_loss is not None:
            total = total + cfg.distill_action_coef * action_loss
        if feature_loss is not None:
            total = total + cfg.distill_feature_coef * feature_loss

    return total
