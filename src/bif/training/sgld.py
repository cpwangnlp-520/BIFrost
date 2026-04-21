"""Localized SGLD sampler for Bayesian Influence Function."""

from __future__ import annotations

import math

import torch
from torch import nn

from bif.config import SGLDConfig
from bif.training.loss import per_example_causal_lm_loss


class LocalizedSGLDSampler:
    """SGLD sampler anchored to a reference parameter set.

    The sampler perturbs model parameters around an anchor point using
    stochastic gradient updates with Gaussian noise, enabling posterior
    sampling for influence estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        anchor_params: dict[str, torch.Tensor],
        config: SGLDConfig,
        source_dataset_size: int,
    ):
        self.model = model
        self.anchor_params = anchor_params
        self.cfg = config
        self.source_dataset_size = source_dataset_size
        self.params: list[tuple[str, nn.Parameter]] = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        if not self.params:
            raise ValueError("No trainable parameters for SGLD updates")

    def reset_to_anchor(self) -> None:
        """Reset all trainable parameters to anchor values."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.anchor_params[name])

    def step(
        self,
        batch: dict[str, torch.Tensor],
        step_generator: torch.Generator | None = None,
    ) -> float:
        """Execute one SGLD update step.

        Args:
            batch: Input batch with input_ids, attention_mask, labels.
            step_generator: Optional RNG for reproducible noise.

        Returns:
            Mean loss value for the batch.
        """
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        per_ex_loss = per_example_causal_lm_loss(
            labels=batch["labels"], logits=outputs.logits
        )
        batch_mean_loss = per_ex_loss.mean()
        batch_mean_loss.backward()

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        lr = self.cfg.lr
        gamma = self.cfg.gamma
        scale = self.cfg.beta * float(self.source_dataset_size)

        with torch.no_grad():
            for name, p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad.detach() * scale
                if self.cfg.weight_decay != 0.0:
                    grad = grad + self.cfg.weight_decay * p.data
                grad = grad + gamma * (p.data - self.anchor_params[name])
                noise = torch.randn(
                    p.shape,
                    device=p.device,
                    dtype=p.dtype,
                    generator=step_generator,
                ) * (math.sqrt(lr) * self.cfg.noise_scale)
                p.add_(-0.5 * lr * grad + noise)

        return float(batch_mean_loss.detach().item())
