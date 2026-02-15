"""Integration tests for voice cloning pipeline.

These tests verify the full pipeline from loading a model through to
generation. Since model weights in this repo are placeholder stubs,
tests requiring real inference are marked with @pytest.mark.skipif and
will only run when full weights are available.

Tests cover:
- Full pipeline: load model -> encode voice -> generate -> save audio
- Backward compatibility (legacy voice files without semantic_mean)
- Voice reuse across multiple generations
- strip_encoders() preventing voice_prompt usage
- Processor and model interaction
- Config round-tripping
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from kugelaudio_open import (
    KugelAudioConfig,
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)
from kugelaudio_open.models.kugelaudio_model import KugelAudioModel

from tests._helpers import PROJECT_ROOT, VOICES_DIR


# ---------------------------------------------------------------------------
# Check if real model weights are available
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(PROJECT_ROOT, "kugelaudio-0-open")
_has_real_weights = False
if os.path.exists(os.path.join(_MODEL_DIR, "model.safetensors.index.json")):
    # Check if any shard is larger than 1 KB (stub files are ~135 bytes)
    import glob
    shards = glob.glob(os.path.join(_MODEL_DIR, "model-*.safetensors"))
    if shards:
        _has_real_weights = any(os.path.getsize(s) > 10000 for s in shards)

needs_real_weights = pytest.mark.skipif(
    not _has_real_weights,
    reason="Requires real model weights (current weights are stubs)",
)


# ---------------------------------------------------------------------------
# Config and architecture integration tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config loading and round-tripping."""

    def test_config_from_json(self):
        """Config should load correctly from the config.json in the model dir."""
        config_path = os.path.join(_MODEL_DIR, "config.json")
        if os.path.exists(config_path):
            config = KugelAudioConfig.from_pretrained(_MODEL_DIR)
            assert config.acoustic_vae_dim == 64, (
                f"acoustic_vae_dim should be 64, got {config.acoustic_vae_dim}"
            )
            assert config.semantic_vae_dim == 128, (
                f"semantic_vae_dim should be 128, got {config.semantic_vae_dim}"
            )
        else:
            pytest.skip("config.json not found in model directory")

    def test_config_has_semantic_tokenizer(self):
        """Config should include semantic_tokenizer_config."""
        config = KugelAudioConfig()
        assert hasattr(config, "semantic_tokenizer_config"), (
            "Config must have semantic_tokenizer_config"
        )
        assert config.semantic_tokenizer_config is not None

    def test_config_has_acoustic_tokenizer(self):
        """Config should include acoustic_tokenizer_config."""
        config = KugelAudioConfig()
        assert hasattr(config, "acoustic_tokenizer_config"), (
            "Config must have acoustic_tokenizer_config"
        )

    def test_config_derived_dims(self):
        """acoustic_vae_dim and semantic_vae_dim should be derived from sub-configs."""
        config = KugelAudioConfig(
            acoustic_tokenizer_config={"vae_dim": 64},
            semantic_tokenizer_config={"vae_dim": 128},
        )
        assert config.acoustic_vae_dim == 64
        assert config.semantic_vae_dim == 128


class TestModelArchitectureIntegration:
    """Tests for model architecture with small configs."""

    def test_model_has_dual_tokenizers(self, minimal_config):
        """Model should have both acoustic and semantic tokenizers."""
        model = KugelAudioModel(minimal_config)
        assert hasattr(model, "acoustic_tokenizer"), (
            "Model should have acoustic_tokenizer"
        )
        assert hasattr(model, "semantic_tokenizer"), (
            "Model should have semantic_tokenizer"
        )

    def test_model_has_dual_connectors(self, minimal_config):
        """Model should have both acoustic and semantic connectors."""
        model = KugelAudioModel(minimal_config)
        assert hasattr(model, "acoustic_connector"), (
            "Model should have acoustic_connector"
        )
        assert hasattr(model, "semantic_connector"), (
            "Model should have semantic_connector"
        )

    def test_strip_encoders_removes_encoders(self, minimal_config):
        """strip_encoders() should set encoder attributes to None."""
        model = KugelAudioModel(minimal_config)

        # Verify encoders exist before stripping
        assert model.acoustic_tokenizer.encoder is not None, (
            "acoustic encoder should exist before stripping"
        )
        assert model.semantic_tokenizer.encoder is not None, (
            "semantic encoder should exist before stripping"
        )

        model.strip_encoders()

        assert model.acoustic_tokenizer.encoder is None, (
            "acoustic encoder should be None after strip_encoders()"
        )
        assert model.semantic_tokenizer.encoder is None, (
            "semantic encoder should be None after strip_encoders()"
        )

    def test_strip_encoders_keeps_decoder(self, minimal_config):
        """strip_encoders() should keep the acoustic decoder intact."""
        model = KugelAudioModel(minimal_config)
        model.strip_encoders()

        assert hasattr(model.acoustic_tokenizer, "decoder"), (
            "acoustic decoder should remain after strip_encoders()"
        )
        assert model.acoustic_tokenizer.decoder is not None, (
            "acoustic decoder should not be None after strip_encoders()"
        )


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy voice files."""

    def test_legacy_voice_without_semantic_mean(self, voice_cache_legacy):
        """_process_speech_inputs should handle voice_cache without semantic_mean."""
        # This tests the fallback path in _process_speech_inputs where
        # semantic_mean is not present in the voice_cache dict.
        model = MagicMock(spec=KugelAudioForConditionalGenerationInference)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        model.parameters = MagicMock(side_effect=lambda: iter([dummy_param]))
        model.config = MagicMock()
        model.config.acoustic_vae_dim = 64
        model.config.semantic_vae_dim = 128

        model.acoustic_tokenizer = MagicMock()
        model.acoustic_tokenizer.fix_std = torch.tensor(0.5)
        model.speech_scaling_factor = torch.tensor(float("nan"))
        model.speech_bias_factor = torch.tensor(float("nan"))

        model.acoustic_connector = MagicMock(return_value=torch.randn(1, 10, 64))
        model.semantic_connector = MagicMock(return_value=torch.randn(1, 10, 64))

        # Should not raise with legacy voice_cache
        acoustic_features, speech_embeds = (
            KugelAudioForConditionalGenerationInference._process_speech_inputs(
                model,
                speech_tensors=None,
                speech_masks=None,
                voice_cache=voice_cache_legacy,
            )
        )
        assert acoustic_features is not None, (
            "Should handle legacy voice files without semantic_mean"
        )

    def test_legacy_voice_file_loads(self, tmp_legacy_voice_file):
        """Legacy .pt files without semantic_mean should load successfully."""
        cache = KugelAudioForConditionalGenerationInference.load_voice(
            tmp_legacy_voice_file
        )
        assert "acoustic_mean" in cache
        assert "semantic_mean" not in cache, (
            "Legacy file should not contain semantic_mean"
        )

    def test_voice_2d_acoustic_mean(self, voice_cache_2d):
        """voice_cache with 2-D acoustic_mean should be handled by the processor."""
        from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor
        from kugelaudio_open.processors.audio_processor import AudioProcessor

        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(
            side_effect=lambda text, **kw: list(range(len(text.split())))
        )

        processor = KugelAudioProcessor(
            tokenizer=tokenizer,
            audio_processor=AudioProcessor(sampling_rate=24000),
        )

        result = processor(
            text="Hello world",
            voice_cache=voice_cache_2d,
            return_tensors="pt",
        )

        assert "voice_cache" in result, "Should accept 2-D acoustic_mean in voice_cache"
        assert "speech_input_mask" in result
        # With 2-D mean, num_voice_tokens should be derived from dim 0
        mask = result["speech_input_mask"]
        true_count = mask.sum().item()
        assert true_count == 10, (
            f"Expected 10 voice tokens from 2-D mean with shape [10, 64], got {true_count}"
        )


# ---------------------------------------------------------------------------
# Voice reuse tests
# ---------------------------------------------------------------------------


class TestVoiceReuse:
    """Tests for reusing the same voice across multiple generations."""

    def test_voice_cache_can_be_reused(self, voice_cache_full):
        """The same voice_cache should be usable across multiple processor calls."""
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(
            side_effect=lambda text, **kw: list(range(len(text.split())))
        )

        processor = KugelAudioProcessor(
            tokenizer=tokenizer,
            audio_processor=MagicMock(),
        )

        result1 = processor(
            text="Hello", voice_cache=voice_cache_full, return_tensors="pt"
        )
        result2 = processor(
            text="World", voice_cache=voice_cache_full, return_tensors="pt"
        )

        # Both should have voice_cache
        assert "voice_cache" in result1
        assert "voice_cache" in result2

        # The cache object should be the same
        assert result1["voice_cache"] is voice_cache_full
        assert result2["voice_cache"] is voice_cache_full

    def test_load_voice_cache_deterministic(self):
        """Loading the same voice twice should give equal tensors."""
        voices_json_path = os.path.join(VOICES_DIR, "voices.json")
        if not os.path.exists(voices_json_path):
            pytest.skip("voices.json not found")

        import json

        with open(voices_json_path) as f:
            registry = json.load(f)

        processor = KugelAudioProcessor(
            tokenizer=MagicMock(),
            audio_processor=MagicMock(),
            voices_registry=registry,
            voices_dir=VOICES_DIR,
        )

        cache1 = processor.load_voice_cache("radio")
        cache2 = processor.load_voice_cache("radio")

        assert torch.equal(cache1["acoustic_mean"], cache2["acoustic_mean"]), (
            "Loading the same voice twice should yield identical tensors"
        )


# ---------------------------------------------------------------------------
# Processor-model integration tests
# ---------------------------------------------------------------------------


class TestProcessorModelIntegration:
    """Tests verifying processor output is compatible with model input."""

    def test_processor_output_keys_for_voice_cache(self, processor_with_mock_tokenizer):
        """Processor output with voice= should have keys that the model's generate() accepts."""
        result = processor_with_mock_tokenizer(
            text="Hello world", voice="radio", return_tensors="pt"
        )

        # These are the keys that generate() expects
        assert "text_ids" in result, "Must have text_ids"
        assert "speech_input_mask" in result, "Must have speech_input_mask"
        assert "voice_cache" in result, "Must have voice_cache for voice="

    def test_processor_output_keys_for_voice_prompt(self, processor_with_mock_tokenizer, sample_audio_tensor):
        """Processor output with voice_prompt= should have keys for model's generate()."""
        result = processor_with_mock_tokenizer(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )

        assert "text_ids" in result, "Must have text_ids"
        assert "speech_input_mask" in result, "Must have speech_input_mask"
        assert "speech_tensors" in result, "Must have speech_tensors for voice_prompt="
        assert "speech_masks" in result, "Must have speech_masks for voice_prompt="

    def test_text_ids_shape_is_2d(self, processor_with_mock_tokenizer):
        """text_ids should be 2-D [batch, seq_len] with return_tensors='pt'."""
        result = processor_with_mock_tokenizer(
            text="Hello world", return_tensors="pt"
        )
        assert result["text_ids"].dim() == 2, (
            f"text_ids should be 2-D, got {result['text_ids'].dim()}-D"
        )

    def test_speech_input_mask_shape_matches_text_ids(self, processor_with_mock_tokenizer):
        """speech_input_mask should have same shape as text_ids."""
        result = processor_with_mock_tokenizer(
            text="Hello world", voice="radio", return_tensors="pt"
        )
        assert result["speech_input_mask"].shape == result["text_ids"].shape, (
            f"speech_input_mask shape {result['speech_input_mask'].shape} "
            f"does not match text_ids shape {result['text_ids'].shape}"
        )


# ---------------------------------------------------------------------------
# Encode voice prompt with real model (needs weights)
# ---------------------------------------------------------------------------


@needs_real_weights
class TestEncodeVoicePromptRealModel:
    """Integration tests that require real model weights."""

    def test_encode_voice_from_sample_wav(self):
        """Encode a real WAV file through the model's encoders."""
        sample_wav = os.path.join(VOICES_DIR, "samples", "radio_voice.wav")
        if not os.path.exists(sample_wav):
            pytest.skip("Sample WAV not found")

        model = KugelAudioForConditionalGenerationInference.from_pretrained(_MODEL_DIR)

        result = model.encode_voice_prompt(sample_wav)

        assert "acoustic_mean" in result
        assert "semantic_mean" in result
        assert result["acoustic_mean"].shape[-1] == 64
        assert result["semantic_mean"].shape[-1] == 128

    def test_full_pipeline_with_voice(self):
        """Full pipeline: load model -> load processor -> encode voice -> generate."""
        model = KugelAudioForConditionalGenerationInference.from_pretrained(_MODEL_DIR)

        processor = KugelAudioProcessor.from_pretrained(PROJECT_ROOT)

        inputs = processor(text="Hello world", voice="radio", return_tensors="pt")
        output = model.generate(
            text_ids=inputs["text_ids"],
            voice_cache=inputs.get("voice_cache"),
            speech_input_mask=inputs.get("speech_input_mask"),
            max_new_tokens=10,
            show_progress=False,
        )

        assert output.sequences is not None
        assert output.speech_outputs is not None

    def test_full_pipeline_with_voice_prompt(self):
        """Full pipeline with voice_prompt (raw audio cloning)."""
        sample_wav = os.path.join(VOICES_DIR, "samples", "radio_voice.wav")
        if not os.path.exists(sample_wav):
            pytest.skip("Sample WAV not found")

        model = KugelAudioForConditionalGenerationInference.from_pretrained(_MODEL_DIR)

        processor = KugelAudioProcessor.from_pretrained(PROJECT_ROOT)

        inputs = processor(
            text="Hello world", voice_prompt=sample_wav, return_tensors="pt"
        )
        output = model.generate(
            text_ids=inputs["text_ids"],
            speech_tensors=inputs.get("speech_tensors"),
            speech_masks=inputs.get("speech_masks"),
            speech_input_mask=inputs.get("speech_input_mask"),
            max_new_tokens=10,
            show_progress=False,
        )

        assert output.sequences is not None


# ---------------------------------------------------------------------------
# Strip encoders integration test
# ---------------------------------------------------------------------------


class TestStripEncodersIntegration:
    """Tests that strip_encoders prevents voice_prompt but allows voice_cache."""

    def test_stripped_model_rejects_raw_audio(self, minimal_config):
        """After strip_encoders(), encode_voice_prompt() should raise RuntimeError."""
        model = KugelAudioForConditionalGenerationInference(minimal_config)
        model.model.strip_encoders()

        with pytest.raises(RuntimeError, match="stripped"):
            model.encode_voice_prompt(torch.randn(1, 1, 24000))

    def test_stripped_model_accepts_voice_cache(self, minimal_config):
        """After strip_encoders(), _process_speech_inputs() with voice_cache should work."""
        model = KugelAudioForConditionalGenerationInference(minimal_config)
        model.model.strip_encoders()

        voice_cache = {
            "acoustic_mean": torch.randn(1, 5, 64),
            "acoustic_std": torch.tensor(0.5),
            "semantic_mean": torch.randn(1, 5, 128),
        }

        # Should not raise
        acoustic_features, speech_embeds = model._process_speech_inputs(
            speech_tensors=None,
            speech_masks=None,
            voice_cache=voice_cache,
        )
        assert acoustic_features is not None


# ---------------------------------------------------------------------------
# Multilingual generation tests (mock-based)
# ---------------------------------------------------------------------------


class TestMultilingualVoices:
    """Tests for using voices with different language texts."""

    def test_same_voice_different_texts(self, processor_with_mock_tokenizer):
        """Same voice should work with different text inputs."""
        result_en = processor_with_mock_tokenizer(
            text="Hello world", voice="radio", return_tensors="pt"
        )
        result_de = processor_with_mock_tokenizer(
            text="Hallo Welt", voice="radio", return_tensors="pt"
        )

        # Both should have voice_cache
        assert "voice_cache" in result_en
        assert "voice_cache" in result_de

    def test_different_voices_same_text(self, processor_with_mock_tokenizer):
        """Different voices should produce different voice_cache dicts."""
        result_radio = processor_with_mock_tokenizer(
            text="Hello", voice="radio", return_tensors="pt"
        )
        result_angry = processor_with_mock_tokenizer(
            text="Hello", voice="angry", return_tensors="pt"
        )

        assert "voice_cache" in result_radio
        assert "voice_cache" in result_angry
        # The caches should not be the same object
        assert result_radio["voice_cache"] is not result_angry["voice_cache"], (
            "Different voices should produce different cache objects"
        )
