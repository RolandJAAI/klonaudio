"""Shared fixtures for KugelAudio voice cloning tests.

Provides reusable fixtures for model configs, mock models, sample audio,
and pre-encoded voice caches used across test modules.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests._helpers import PROJECT_ROOT, VOICES_DIR, SAMPLES_DIR


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio_tensor():
    """Create a synthetic 24 kHz mono audio tensor (5 seconds)."""
    return torch.randn(1, 1, 24000 * 5)


@pytest.fixture
def short_audio_tensor():
    """Create a very short audio tensor (0.1 seconds at 24 kHz)."""
    return torch.randn(1, 1, 2400)


@pytest.fixture
def sample_audio_1d():
    """Create a 1-D audio tensor (raw waveform, 1 second at 24 kHz)."""
    return torch.randn(24000)


@pytest.fixture
def sample_audio_2d():
    """Create a 2-D audio tensor [batch, samples] (1 second at 24 kHz)."""
    return torch.randn(1, 24000)


# ---------------------------------------------------------------------------
# Voice cache fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_cache_full():
    """A complete voice cache dict with both acoustic and semantic means."""
    return {
        "acoustic_mean": torch.randn(1, 10, 64),
        "acoustic_std": torch.tensor(0.5),
        "semantic_mean": torch.randn(1, 10, 128),
        "audio_length": 24000 * 5,
        "sample_rate": 24000,
    }


@pytest.fixture
def voice_cache_legacy():
    """A legacy voice cache dict without semantic_mean (backward compat)."""
    return {
        "acoustic_mean": torch.randn(1, 10, 64),
        "acoustic_std": torch.tensor(0.5),
    }


@pytest.fixture
def voice_cache_2d():
    """A voice cache where acoustic_mean is 2-D [time, dim]."""
    return {
        "acoustic_mean": torch.randn(10, 64),
        "acoustic_std": torch.tensor(0.5),
        "semantic_mean": torch.randn(10, 128),
    }


# ---------------------------------------------------------------------------
# Temporary file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_voice_file(voice_cache_full, tmp_path):
    """Save a voice cache dict to a temporary .pt file and return its path."""
    path = tmp_path / "test_voice.pt"
    torch.save(voice_cache_full, str(path))
    return str(path)


@pytest.fixture
def tmp_legacy_voice_file(voice_cache_legacy, tmp_path):
    """Save a legacy voice cache (no semantic_mean) to a temporary .pt file."""
    path = tmp_path / "legacy_voice.pt"
    torch.save(voice_cache_legacy, str(path))
    return str(path)


@pytest.fixture
def tmp_wav_file(tmp_path):
    """Create a minimal WAV file for testing."""
    import struct
    import wave

    wav_path = tmp_path / "test_audio.wav"
    sample_rate = 24000
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)
    # Generate silence (zeros)
    samples = [0] * n_samples

    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        for s in samples:
            f.writeframesraw(struct.pack("<h", s))

    return str(wav_path)


# ---------------------------------------------------------------------------
# Processor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that returns predictable token IDs."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(side_effect=lambda text, **kw: list(range(len(text.split()))))
    return tokenizer


@pytest.fixture
def voices_registry():
    """Return the standard voices registry dict."""
    return {
        "default": {
            "file": "default.pt",
            "description": "Default neutral voice",
            "language": "en",
            "gender": "neutral",
            "source": "generated",
            "encoded_with": "kugelaudio-0-open",
        },
        "warm": {
            "file": "warm.pt",
            "description": "Warm, friendly voice",
            "language": "en",
            "gender": "neutral",
            "source": "generated",
            "encoded_with": "kugelaudio-0-open",
        },
        "clear": {
            "file": "clear.pt",
            "description": "Clear, professional voice",
            "language": "en",
            "gender": "neutral",
            "source": "generated",
            "encoded_with": "kugelaudio-0-open",
        },
    }


@pytest.fixture
def processor_with_mock_tokenizer(mock_tokenizer, voices_registry):
    """Create a KugelAudioProcessor with a mocked tokenizer for unit tests."""
    from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor
    from kugelaudio_open.processors.audio_processor import AudioProcessor

    return KugelAudioProcessor(
        tokenizer=mock_tokenizer,
        audio_processor=AudioProcessor(sampling_rate=24000),
        voices_registry=voices_registry,
        voices_dir=VOICES_DIR,
        model_name_or_path=PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kugelaudio_config():
    """Create a KugelAudioConfig with default settings."""
    from kugelaudio_open.configs import KugelAudioConfig

    return KugelAudioConfig()


@pytest.fixture
def minimal_config():
    """Create a minimal KugelAudioConfig for fast model instantiation.

    Uses a very small decoder to speed up tests that need a real model object
    but don't need production-size weights.
    """
    from kugelaudio_open.configs import KugelAudioConfig

    return KugelAudioConfig(
        decoder_config={
            "model_type": "qwen2",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "vocab_size": 152064,
        },
        acoustic_tokenizer_config={
            "vae_dim": 64,
            "encoder_n_filters": 8,
            "decoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "decoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
        },
        semantic_tokenizer_config={
            "vae_dim": 128,
            "encoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
        },
        diffusion_head_config={
            "hidden_size": 64,
            "head_layers": 1,
            "latent_size": 64,
            "speech_vae_dim": 64,
        },
    )
