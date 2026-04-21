"""Smoke tests to verify compatibility with transformers >= 5.0."""
from __future__ import annotations

import inspect
import warnings

import pytest
import torch
import transformers


def test_transformers_version():
    """Ensure transformers >= 5.0 is installed."""
    major = int(transformers.__version__.split(".")[0])
    assert major >= 5, f"Expected transformers >= 5.0, got {transformers.__version__}"


def test_gradscaler_new_api():
    """torch.amp.GradScaler('cuda', ...) should not raise FutureWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="torch")
        # Should not raise — new API
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert not scaler.is_enabled()


def test_gradscaler_old_api_warns():
    """torch.cuda.amp.GradScaler raises FutureWarning (deprecated in torch 2.x)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.amp.GradScaler(enabled=False)
    future_warns = [x for x in w if issubclass(x.category, FutureWarning)]
    assert future_warns, "Expected FutureWarning from deprecated torch.cuda.amp.GradScaler"


def test_per_example_causal_lm_loss_signature():
    """Confirm per_example_causal_lm_loss(labels, logits) order and output shape."""
    from bif.training.loss import per_example_causal_lm_loss

    sig = inspect.signature(per_example_causal_lm_loss)
    param_names = list(sig.parameters.keys())
    assert param_names[0] == "labels", f"First param should be 'labels', got '{param_names[0]}'"
    assert param_names[1] == "logits", f"Second param should be 'logits', got '{param_names[1]}'"


def test_per_example_causal_lm_loss_output():
    """per_example_causal_lm_loss returns per-example scalar losses."""
    from bif.training.loss import per_example_causal_lm_loss

    B, T, V = 4, 16, 64
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    loss = per_example_causal_lm_loss(labels, logits)
    assert loss.shape == (B,), f"Expected shape ({B},), got {loss.shape}"
    assert loss.dtype == torch.float32
    assert (loss >= 0).all(), "Losses should be non-negative"


def test_trainer_processing_class():
    """Trainer should accept processing_class (not tokenizer) in transformers 5.x."""
    from transformers import Trainer

    sig = inspect.signature(Trainer.__init__)
    assert "processing_class" in sig.parameters, (
        "Trainer.__init__ missing 'processing_class' — upgrade to transformers >= 5.0"
    )
    assert "tokenizer" not in sig.parameters, (
        "Trainer.__init__ still has deprecated 'tokenizer' param"
    )


def test_training_arguments_use_cpu():
    """TrainingArguments should have use_cpu (no_cuda was renamed in transformers 4.x)."""
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    assert "use_cpu" in sig.parameters, "TrainingArguments missing 'use_cpu'"


def test_schedule_trainer_imports():
    """schedule_trainer.py must import cleanly with no FutureWarning from torch."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="torch")
        import bif.training.schedule_trainer  # noqa: F401


def test_checkpoint_trainer_imports():
    """checkpoint_trainer.py must import cleanly."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="torch")
        import bif.training.checkpoint_trainer  # noqa: F401


def test_bif_runner_imports():
    """bif_runner.py must import cleanly."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="torch")
        import bif.analysis.bif_runner  # noqa: F401
