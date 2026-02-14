"""Tests for the encode_voice_prompt() method on the inference model.

Tests cover:
- Encoding from a WAV file path
- Encoding from a raw torch.Tensor (various shapes)
- Returned dict structure and required keys
- Tensor shapes matching config dimensions (acoustic_vae_dim=64, semantic_vae_dim=128)
- Error handling when encoders have been stripped
- Audio length and sample rate handling
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from kugelaudio_open.models.kugelaudio_inference import (
    KugelAudioForConditionalGenerationInference,
)

from tests._helpers import make_mock_model_base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(acoustic_vae_dim=64, semantic_vae_dim=128, encoder_present=True):
    """Build a lightweight mock of KugelAudioForConditionalGenerationInference.

    Instead of instantiating the real model (which requires large weights), we
    create a MagicMock with the attributes that encode_voice_prompt() relies on.
    """
    return make_mock_model_base(
        acoustic_vae_dim=acoustic_vae_dim,
        semantic_vae_dim=semantic_vae_dim,
        encoder_present=encoder_present,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEncodeVoicePromptStructure:
    """Tests for the returned dictionary structure of encode_voice_prompt()."""

    def test_returned_dict_has_required_keys(self, sample_audio_tensor):
        """encode_voice_prompt() must return dict with all expected keys."""
        model = _make_mock_model()

        # Call the real method, bound to our mock
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )

        expected_keys = {
            "acoustic_mean",
            "acoustic_std",
            "semantic_mean",
            "audio_length",
            "sample_rate",
        }
        assert set(result.keys()) == expected_keys, (
            f"Returned keys {set(result.keys())} do not match expected {expected_keys}"
        )

    def test_acoustic_mean_is_tensor(self, sample_audio_tensor):
        """acoustic_mean in the result must be a torch.Tensor."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert isinstance(result["acoustic_mean"], torch.Tensor), (
            "acoustic_mean should be a torch.Tensor"
        )

    def test_semantic_mean_is_tensor(self, sample_audio_tensor):
        """semantic_mean in the result must be a torch.Tensor."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert isinstance(result["semantic_mean"], torch.Tensor), (
            "semantic_mean should be a torch.Tensor"
        )

    def test_audio_length_matches_input(self, sample_audio_tensor):
        """audio_length should match the number of samples in the input tensor."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        expected_length = sample_audio_tensor.shape[-1]
        assert result["audio_length"] == expected_length, (
            f"audio_length {result['audio_length']} != expected {expected_length}"
        )

    def test_sample_rate_default(self, sample_audio_tensor):
        """Default sample_rate should be 24000."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["sample_rate"] == 24000, (
            f"Default sample_rate should be 24000, got {result['sample_rate']}"
        )

    def test_sample_rate_custom(self, sample_audio_tensor):
        """Custom sample_rate should be preserved in the result."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor, sample_rate=16000
        )
        assert result["sample_rate"] == 16000, (
            f"Custom sample_rate should be 16000, got {result['sample_rate']}"
        )


class TestEncodeVoicePromptTensorShapes:
    """Tests verifying that output tensor shapes match config dimensions."""

    def test_acoustic_mean_last_dim_matches_vae_dim(self, sample_audio_tensor):
        """acoustic_mean last dimension must equal acoustic_vae_dim (64)."""
        acoustic_dim = 64
        model = _make_mock_model(acoustic_vae_dim=acoustic_dim)
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["acoustic_mean"].shape[-1] == acoustic_dim, (
            f"acoustic_mean last dim {result['acoustic_mean'].shape[-1]} "
            f"does not match acoustic_vae_dim={acoustic_dim}"
        )

    def test_semantic_mean_last_dim_matches_vae_dim(self, sample_audio_tensor):
        """semantic_mean last dimension must equal semantic_vae_dim (128)."""
        semantic_dim = 128
        model = _make_mock_model(semantic_vae_dim=semantic_dim)
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["semantic_mean"].shape[-1] == semantic_dim, (
            f"semantic_mean last dim {result['semantic_mean'].shape[-1]} "
            f"does not match semantic_vae_dim={semantic_dim}"
        )

    def test_acoustic_mean_is_3d(self, sample_audio_tensor):
        """acoustic_mean should be a 3-D tensor [batch, time, dim]."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["acoustic_mean"].dim() == 3, (
            f"acoustic_mean should be 3-D, got {result['acoustic_mean'].dim()}-D"
        )

    def test_semantic_mean_is_3d(self, sample_audio_tensor):
        """semantic_mean should be a 3-D tensor [batch, time, dim]."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["semantic_mean"].dim() == 3, (
            f"semantic_mean should be 3-D, got {result['semantic_mean'].dim()}-D"
        )


class TestEncodeVoicePromptInputHandling:
    """Tests for different input types and shapes."""

    def test_1d_tensor_input(self, sample_audio_1d):
        """1-D tensor (raw waveform) should be auto-expanded to [1, 1, T]."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_1d
        )
        assert "acoustic_mean" in result, "Should successfully encode 1-D tensor"
        # Verify the audio was reshaped before encoding
        call_args = model.acoustic_tokenizer.encode.call_args
        encoded_audio = call_args[0][0]
        assert encoded_audio.dim() == 3, (
            f"Audio passed to encoder should be 3-D, got {encoded_audio.dim()}-D"
        )

    def test_2d_tensor_input(self, sample_audio_2d):
        """2-D tensor [batch, samples] should be auto-expanded to [batch, 1, T]."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_2d
        )
        assert "acoustic_mean" in result, "Should successfully encode 2-D tensor"
        call_args = model.acoustic_tokenizer.encode.call_args
        encoded_audio = call_args[0][0]
        assert encoded_audio.dim() == 3, (
            f"Audio passed to encoder should be 3-D, got {encoded_audio.dim()}-D"
        )

    def test_3d_tensor_input(self, sample_audio_tensor):
        """3-D tensor [batch, channels, samples] should pass through directly."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert "acoustic_mean" in result, "Should successfully encode 3-D tensor"

    def test_file_path_input(self, tmp_wav_file):
        """String file path should trigger audio loading and encoding."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=tmp_wav_file
        )
        assert "acoustic_mean" in result, "Should successfully encode from WAV file"
        assert result["audio_length"] > 0, "audio_length should be positive for file input"

    def test_both_encoders_are_called(self, sample_audio_tensor):
        """Both acoustic and semantic tokenizer encode() methods must be called."""
        model = _make_mock_model()
        KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        model.acoustic_tokenizer.encode.assert_called_once()
        model.semantic_tokenizer.encode.assert_called_once()


class TestEncodeVoicePromptErrors:
    """Tests for error conditions in encode_voice_prompt()."""

    def test_stripped_encoder_raises_runtime_error(self, sample_audio_tensor):
        """Should raise RuntimeError when acoustic encoder is None (stripped)."""
        model = _make_mock_model(encoder_present=False)

        with pytest.raises(RuntimeError, match="stripped"):
            KugelAudioForConditionalGenerationInference.encode_voice_prompt(
                model, voice_audio=sample_audio_tensor
            )

    def test_nonexistent_file_raises_error(self):
        """Should raise an error when given a non-existent file path."""
        model = _make_mock_model()
        with pytest.raises(Exception):
            KugelAudioForConditionalGenerationInference.encode_voice_prompt(
                model, voice_audio="/nonexistent/path/to/audio.wav"
            )


class TestEncodeVoicePromptCPUOutput:
    """Tests that outputs are placed on CPU (for caching/saving)."""

    def test_acoustic_mean_on_cpu(self, sample_audio_tensor):
        """acoustic_mean should be on CPU for easy serialization."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["acoustic_mean"].device == torch.device("cpu"), (
            "acoustic_mean should be on CPU"
        )

    def test_semantic_mean_on_cpu(self, sample_audio_tensor):
        """semantic_mean should be on CPU for easy serialization."""
        model = _make_mock_model()
        result = KugelAudioForConditionalGenerationInference.encode_voice_prompt(
            model, voice_audio=sample_audio_tensor
        )
        assert result["semantic_mean"].device == torch.device("cpu"), (
            "semantic_mean should be on CPU"
        )
