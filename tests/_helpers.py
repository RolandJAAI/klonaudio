"""Shared test helpers for KugelAudio tests.

This module contains constants and factory functions shared across test files.
Unlike conftest.py (which pytest auto-discovers for fixtures), this module is
imported explicitly by test modules that need it.
"""

import os
from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(PROJECT_ROOT, "voices")
SAMPLES_DIR = os.path.join(VOICES_DIR, "samples")


# ---------------------------------------------------------------------------
# Mock model factory
# ---------------------------------------------------------------------------


def make_mock_model_base(
    acoustic_vae_dim=64,
    semantic_vae_dim=128,
    encoder_present=True,
):
    """Build a base mock of KugelAudioForConditionalGenerationInference.

    Creates a MagicMock with the shared attributes that both
    encode_voice_prompt() and generate() rely on: dummy parameters,
    acoustic/semantic tokenizers with optional encoders, and encode methods.

    Callers can extend the returned mock with additional attributes
    (config, connectors, etc.) for their specific test needs.
    """
    from kugelaudio_open.models.kugelaudio_inference import (
        KugelAudioForConditionalGenerationInference,
    )

    model = MagicMock(spec=KugelAudioForConditionalGenerationInference)

    # Make parameters() return a fresh iterator each time so that
    # next(self.parameters()) works for both device and dtype calls.
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    model.parameters = MagicMock(side_effect=lambda: iter([dummy_param]))

    # Acoustic tokenizer mock
    acoustic_tokenizer = MagicMock()
    if not encoder_present:
        acoustic_tokenizer.encoder = None
    else:
        acoustic_tokenizer.encoder = MagicMock()
    acoustic_tokenizer.fix_std = torch.tensor(0.5)

    # Acoustic encoder returns an output with mean and std
    acoustic_output = MagicMock()
    acoustic_output.mean = torch.randn(1, 7, acoustic_vae_dim)
    acoustic_output.std = torch.tensor(0.5)
    acoustic_tokenizer.encode = MagicMock(return_value=acoustic_output)

    # Semantic tokenizer mock
    semantic_tokenizer = MagicMock()
    if not encoder_present:
        semantic_tokenizer.encoder = None
    else:
        semantic_tokenizer.encoder = MagicMock()

    semantic_output = MagicMock()
    semantic_output.mean = torch.randn(1, 7, semantic_vae_dim)
    semantic_tokenizer.encode = MagicMock(return_value=semantic_output)

    model.acoustic_tokenizer = acoustic_tokenizer
    model.semantic_tokenizer = semantic_tokenizer

    return model
