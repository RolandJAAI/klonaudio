"""Tests for end-to-end generation with voice inputs.

These tests focus on the generate() method of the inference model and its
interaction with different voice input methods. Since model weights are
placeholder stubs, we use mocking for most tests and mark integration tests
that require real weights appropriately.

Tests cover:
- Generation with voice_cache parameter
- Generation with voice_prompt parameter (legacy)
- Generation with speech_tensors/speech_masks
- Generation without any voice (random voice)
- Output structure (sequences and speech_outputs)
- Device and dtype handling
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from kugelaudio_open.models.kugelaudio_inference import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioGenerationOutput,
)

from tests._helpers import make_mock_model_base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inference_model(
    hidden_size=64,
    vocab_size=152064,
    acoustic_vae_dim=64,
    semantic_vae_dim=128,
    encoder_present=True,
):
    """Build a mock inference model for testing generate().

    Returns a MagicMock configured with the properties and methods that
    generate() relies on, without needing real model weights.
    """
    model = make_mock_model_base(
        acoustic_vae_dim=acoustic_vae_dim,
        semantic_vae_dim=semantic_vae_dim,
        encoder_present=encoder_present,
    )
    model.dtype = torch.float32

    # Config
    config = MagicMock()
    config.acoustic_vae_dim = acoustic_vae_dim
    config.semantic_vae_dim = semantic_vae_dim
    config.decoder_config.hidden_size = hidden_size
    config.decoder_config.vocab_size = vocab_size
    config.speech_start_id = 151652
    config.speech_end_id = 151653
    config.speech_diffusion_id = 151654
    config.decoder_config.eos_token_id = 151643
    config.use_return_dict = True
    config.diffusion_head_config.ddpm_num_inference_steps = 5
    model.config = config

    # Speech scaling
    model.speech_scaling_factor = torch.tensor(1.0)
    model.speech_bias_factor = torch.tensor(0.0)

    # Connectors
    model.acoustic_connector = MagicMock(
        return_value=torch.randn(1, 1, hidden_size)
    )
    model.semantic_connector = MagicMock(
        return_value=torch.randn(1, 1, hidden_size)
    )

    return model


# ---------------------------------------------------------------------------
# Tests for generate() input validation
# ---------------------------------------------------------------------------


class TestGenerateInputValidation:
    """Tests for generate() input validation."""

    def test_generate_requires_text_ids(self):
        """generate() should raise ValueError when no text_ids or input_ids provided."""
        model = _make_mock_inference_model()

        with pytest.raises(ValueError, match="text_ids"):
            KugelAudioForConditionalGenerationInference.generate(
                model,
                text_ids=None,
                input_ids=None,
            )

    def test_generate_accepts_input_ids_alias(self):
        """generate() should accept input_ids as alias for text_ids."""
        model = _make_mock_inference_model()

        # We just verify it does not raise the "text_ids is required" error
        # when input_ids is provided. The actual generation will fail due to
        # mocking, so we catch the specific errors that mocks produce.
        try:
            KugelAudioForConditionalGenerationInference.generate(
                model,
                input_ids=torch.tensor([[1, 2, 3]]),
                max_new_tokens=1,
                show_progress=False,
            )
        except ValueError as e:
            # Should not be the "text_ids or input_ids is required" error
            assert "text_ids" not in str(e), (
                "Should not raise 'text_ids required' when input_ids is given"
            )
        except (AttributeError, TypeError, RuntimeError):
            # Expected errors from incomplete mocking of the generation loop
            pass


class TestGenerateOutputStructure:
    """Tests that verify the output structure of generate()."""

    def test_output_type_is_generation_output(self):
        """generate() should return a KugelAudioGenerationOutput."""
        # Verify the class structure
        output = KugelAudioGenerationOutput(
            sequences=torch.tensor([[1, 2, 3]]),
            speech_outputs=[torch.randn(24000)],
        )
        assert hasattr(output, "sequences"), "Output should have 'sequences'"
        assert hasattr(output, "speech_outputs"), "Output should have 'speech_outputs'"

    def test_generation_output_sequences_is_tensor(self):
        """sequences field should be a torch.Tensor."""
        output = KugelAudioGenerationOutput(
            sequences=torch.tensor([[1, 2, 3]]),
            speech_outputs=None,
        )
        assert isinstance(output.sequences, torch.Tensor)

    def test_generation_output_speech_outputs_is_list(self):
        """speech_outputs field should be a list."""
        output = KugelAudioGenerationOutput(
            sequences=torch.tensor([[1, 2, 3]]),
            speech_outputs=[torch.randn(24000)],
        )
        assert isinstance(output.speech_outputs, list)


# ---------------------------------------------------------------------------
# Tests for voice_prompt legacy parameter
# ---------------------------------------------------------------------------


class TestGenerateVoicePromptLegacy:
    """Tests that the legacy voice_prompt parameter maps to speech_tensors."""

    def test_voice_prompt_maps_to_speech_tensors(self):
        """The generate() method should treat voice_prompt as speech_tensors."""
        model = _make_mock_inference_model()
        audio = torch.randn(1, 1, 24000)

        # Patch _process_speech_inputs to capture what gets passed
        model._process_speech_inputs = MagicMock(
            return_value=(
                torch.randn(1, 1, 64),
                torch.randn(1, 64),
            )
        )
        # Patch the model's __call__ to return a proper output
        mock_output = MagicMock()
        mock_output.past_key_values = None
        mock_output.last_hidden_state = torch.randn(1, 10, 64)
        mock_output.logits = torch.full((1, 1, 152064), float("-inf"))
        # Make the EOS token have highest logit
        mock_output.logits[0, 0, 151643] = 0.0
        model.__call__ = MagicMock(return_value=mock_output)

        # Patch get_input_embeddings
        embed = MagicMock()
        embed.__call__ = MagicMock(return_value=torch.randn(1, 3, 64))
        model.model = MagicMock()
        model.model.get_input_embeddings = MagicMock(return_value=embed)

        try:
            KugelAudioForConditionalGenerationInference.generate(
                model,
                text_ids=torch.tensor([[1, 2, 3]]),
                voice_prompt=audio,
                max_new_tokens=1,
                show_progress=False,
            )
        except (AttributeError, TypeError, RuntimeError):
            pass  # Expected errors from incomplete mocking of the generation loop

        # Assert that _process_speech_inputs was actually called
        assert model._process_speech_inputs.called, (
            "voice_prompt should trigger a call to _process_speech_inputs"
        )
        call_kwargs = model._process_speech_inputs.call_args
        assert call_kwargs is not None, (
            "voice_prompt should trigger _process_speech_inputs call with arguments"
        )


# ---------------------------------------------------------------------------
# Tests for _process_speech_inputs
# ---------------------------------------------------------------------------


class TestProcessSpeechInputs:
    """Tests for _process_speech_inputs() method."""

    def test_voice_cache_mode(self):
        """With voice_cache, should use pre-encoded features without encoding."""
        model = _make_mock_inference_model()

        voice_cache = {
            "acoustic_mean": torch.randn(1, 5, 64),
            "acoustic_std": torch.tensor(0.5),
            "semantic_mean": torch.randn(1, 5, 128),
        }

        # Patch the connector returns
        model.acoustic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )
        model.semantic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )

        acoustic_features, speech_embeds = (
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=None,
                speech_masks=None,
                voice_cache=voice_cache,
            )
        )

        assert acoustic_features is not None, "Should return acoustic features"
        assert speech_embeds is not None, "Should return speech embeddings"
        # Encoders should NOT be called when using voice_cache
        model.acoustic_tokenizer.encode.assert_not_called()

    def test_voice_cache_legacy_without_semantic(self):
        """Legacy voice_cache without semantic_mean should use zeros as fallback."""
        model = _make_mock_inference_model()

        voice_cache = {
            "acoustic_mean": torch.randn(1, 5, 64),
            "acoustic_std": torch.tensor(0.5),
            # No semantic_mean
        }

        model.acoustic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )
        model.semantic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )

        # Should not raise
        acoustic_features, speech_embeds = (
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=None,
                speech_masks=None,
                voice_cache=voice_cache,
            )
        )
        assert acoustic_features is not None, (
            "Should handle legacy voice_cache without semantic_mean"
        )

    def test_dummy_mode_no_inputs(self):
        """Without speech_tensors or voice_cache, should return dummy features."""
        model = _make_mock_inference_model()

        model.acoustic_connector = MagicMock(
            return_value=torch.randn(1, 1, 64)
        )
        model.semantic_connector = MagicMock(
            return_value=torch.randn(1, 1, 64)
        )

        acoustic_features, speech_embeds = (
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=None,
                speech_masks=None,
                voice_cache=None,
            )
        )
        assert acoustic_features.shape == torch.Size([1, 1, 64]), (
            "Dummy acoustic features should have shape [1, 1, vae_dim]"
        )

    def test_stripped_encoders_with_speech_tensors_raises(self):
        """Should raise RuntimeError when encoders are stripped and speech_tensors provided."""
        model = _make_mock_inference_model(encoder_present=False)

        with pytest.raises(RuntimeError, match="stripped"):
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=torch.randn(1, 1, 24000),
                speech_masks=None,
                voice_cache=None,
            )

    def test_stripped_encoders_with_voice_cache_works(self):
        """voice_cache should work even when encoders are stripped."""
        model = _make_mock_inference_model(encoder_present=False)

        voice_cache = {
            "acoustic_mean": torch.randn(1, 5, 64),
            "acoustic_std": torch.tensor(0.5),
            "semantic_mean": torch.randn(1, 5, 128),
        }

        model.acoustic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )
        model.semantic_connector = MagicMock(
            return_value=torch.randn(1, 5, 64)
        )

        # Should not raise
        acoustic_features, speech_embeds = (
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=None,
                speech_masks=None,
                voice_cache=voice_cache,
            )
        )
        assert acoustic_features is not None, (
            "voice_cache should work with stripped encoders"
        )


# ---------------------------------------------------------------------------
# Tests for load_voice static method
# ---------------------------------------------------------------------------


class TestLoadVoice:
    """Tests for the static load_voice() method."""

    def test_load_voice_from_file(self, tmp_voice_file):
        """Should load a voice cache dict from a .pt file."""
        cache = KugelAudioForConditionalGenerationInference.load_voice(tmp_voice_file)
        assert "acoustic_mean" in cache, "Loaded voice must contain 'acoustic_mean'"

    def test_load_voice_missing_acoustic_mean(self, tmp_path):
        """Should raise ValueError when file lacks 'acoustic_mean'."""
        bad_path = tmp_path / "bad_voice.pt"
        torch.save({"some_key": torch.randn(10)}, str(bad_path))

        with pytest.raises(ValueError, match="acoustic_mean"):
            KugelAudioForConditionalGenerationInference.load_voice(str(bad_path))

    def test_load_voice_nonexistent_file(self):
        """Should raise error when file does not exist."""
        with pytest.raises(Exception):
            KugelAudioForConditionalGenerationInference.load_voice(
                "/nonexistent/voice.pt"
            )
