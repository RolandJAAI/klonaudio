"""Tests for KugelAudioProcessor voice handling.

Tests cover:
- voice_prompt parameter (str path and torch.Tensor)
- voice parameter (named voice from registry)
- voice_cache parameter (pre-encoded dict)
- Parameter priority (voice_cache > voice > voice_prompt)
- Error handling (file not found, unknown voice)
- speech_tensors and speech_masks creation
- load_voice_cache() method
- get_available_voices() method
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor
from kugelaudio_open.processors.audio_processor import AudioProcessor

from tests._helpers import PROJECT_ROOT, VOICES_DIR


# ---------------------------------------------------------------------------
# Fixtures specific to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def processor(mock_tokenizer, voices_registry):
    """Processor with mocked tokenizer and real voices registry."""
    return KugelAudioProcessor(
        tokenizer=mock_tokenizer,
        audio_processor=AudioProcessor(sampling_rate=24000),
        voices_registry=voices_registry,
        voices_dir=VOICES_DIR,
        model_name_or_path=PROJECT_ROOT,
    )


@pytest.fixture
def processor_no_voices(mock_tokenizer):
    """Processor with no voices registry configured."""
    return KugelAudioProcessor(
        tokenizer=mock_tokenizer,
        audio_processor=AudioProcessor(sampling_rate=24000),
        voices_registry={},
        voices_dir=None,
    )


# ---------------------------------------------------------------------------
# get_available_voices() tests
# ---------------------------------------------------------------------------


class TestGetAvailableVoices:
    """Tests for get_available_voices() method."""

    def test_returns_list(self, processor):
        """get_available_voices() should return a list."""
        voices = processor.get_available_voices()
        assert isinstance(voices, list), "Should return a list"

    def test_default_voices_present(self, processor):
        """All three default voices should be available."""
        voices = processor.get_available_voices()
        for name in ["default", "warm", "clear"]:
            assert name in voices, f"Voice '{name}' should be available"

    def test_empty_registry_returns_empty(self, processor_no_voices):
        """Empty registry should return empty list."""
        voices = processor_no_voices.get_available_voices()
        assert voices == [], "Empty registry should return empty list"


# ---------------------------------------------------------------------------
# load_voice_cache() tests
# ---------------------------------------------------------------------------


class TestLoadVoiceCache:
    """Tests for load_voice_cache() method."""

    def test_load_default_voice(self, processor):
        """Should successfully load the 'default' voice."""
        cache = processor.load_voice_cache("default")
        assert "acoustic_mean" in cache, "Loaded voice must contain 'acoustic_mean'"
        assert isinstance(cache["acoustic_mean"], torch.Tensor), (
            "acoustic_mean should be a tensor"
        )

    def test_load_warm_voice(self, processor):
        """Should successfully load the 'warm' voice."""
        cache = processor.load_voice_cache("warm")
        assert "acoustic_mean" in cache, "Loaded voice must contain 'acoustic_mean'"

    def test_load_clear_voice(self, processor):
        """Should successfully load the 'clear' voice."""
        cache = processor.load_voice_cache("clear")
        assert "acoustic_mean" in cache, "Loaded voice must contain 'acoustic_mean'"

    def test_unknown_voice_raises_error(self, processor):
        """Should raise ValueError for unknown voice names."""
        with pytest.raises(ValueError, match="not found"):
            processor.load_voice_cache("nonexistent_voice")

    def test_unknown_voice_shows_available(self, processor):
        """Error message for unknown voice should list available voices."""
        with pytest.raises(ValueError, match="default"):
            processor.load_voice_cache("nonexistent_voice")


# ---------------------------------------------------------------------------
# __call__ with voice parameter (named voice) tests
# ---------------------------------------------------------------------------


class TestProcessorVoiceParam:
    """Tests for using voice= (named voice) parameter."""

    def test_voice_default_returns_voice_cache(self, processor):
        """Using voice='default' should return voice_cache in the result."""
        result = processor(text="Hello world", voice="default", return_tensors="pt")
        assert "voice_cache" in result, (
            "Result should contain 'voice_cache' when voice= is used"
        )
        assert "acoustic_mean" in result["voice_cache"], (
            "voice_cache should contain 'acoustic_mean'"
        )

    def test_voice_does_not_return_speech_tensors(self, processor):
        """Using voice= should NOT return speech_tensors (it returns voice_cache instead)."""
        result = processor(text="Hello world", voice="default", return_tensors="pt")
        assert "speech_tensors" not in result, (
            "voice= should not produce speech_tensors"
        )

    def test_voice_produces_text_ids(self, processor):
        """Result should always contain text_ids."""
        result = processor(text="Hello world", voice="default", return_tensors="pt")
        assert "text_ids" in result, "Result must contain 'text_ids'"
        assert isinstance(result["text_ids"], torch.Tensor), (
            "text_ids should be a tensor when return_tensors='pt'"
        )

    def test_voice_produces_speech_input_mask(self, processor):
        """Result should always contain speech_input_mask."""
        result = processor(text="Hello world", voice="default", return_tensors="pt")
        assert "speech_input_mask" in result, "Result must contain 'speech_input_mask'"
        assert isinstance(result["speech_input_mask"], torch.Tensor)

    def test_speech_input_mask_has_true_values(self, processor):
        """speech_input_mask should have True values where voice tokens are placed."""
        result = processor(text="Hello world", voice="default", return_tensors="pt")
        mask = result["speech_input_mask"]
        assert mask.any(), "speech_input_mask should have at least one True value for voice tokens"

    def test_unknown_voice_name_raises_error(self, processor):
        """Using an unknown voice name should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            processor(text="Hello world", voice="nonexistent_voice", return_tensors="pt")


# ---------------------------------------------------------------------------
# __call__ with voice_cache parameter tests
# ---------------------------------------------------------------------------


class TestProcessorVoiceCacheParam:
    """Tests for using voice_cache= (pre-encoded dict) parameter."""

    def test_voice_cache_passthrough(self, processor, voice_cache_full):
        """voice_cache dict should be passed through to the result."""
        result = processor(
            text="Hello world", voice_cache=voice_cache_full, return_tensors="pt"
        )
        assert "voice_cache" in result, "Result should contain 'voice_cache'"
        assert result["voice_cache"] is voice_cache_full, (
            "voice_cache should be the same dict object"
        )

    def test_voice_cache_no_speech_tensors(self, processor, voice_cache_full):
        """voice_cache should NOT produce speech_tensors."""
        result = processor(
            text="Hello world", voice_cache=voice_cache_full, return_tensors="pt"
        )
        assert "speech_tensors" not in result, (
            "voice_cache should not produce speech_tensors"
        )


# ---------------------------------------------------------------------------
# __call__ with voice_prompt parameter tests
# ---------------------------------------------------------------------------


class TestProcessorVoicePromptParam:
    """Tests for using voice_prompt= (raw audio) parameter."""

    def test_voice_prompt_tensor_returns_speech_tensors(self, processor, sample_audio_tensor):
        """voice_prompt as tensor should produce speech_tensors and speech_masks."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        assert "speech_tensors" in result, "Result should contain 'speech_tensors'"
        assert "speech_masks" in result, "Result should contain 'speech_masks'"

    def test_voice_prompt_tensor_no_voice_cache(self, processor, sample_audio_tensor):
        """voice_prompt should NOT produce voice_cache in the result."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        assert "voice_cache" not in result, "voice_prompt should not produce voice_cache"

    def test_voice_prompt_1d_tensor(self, processor, sample_audio_1d):
        """1-D tensor voice_prompt should be expanded to [1, 1, T]."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_1d,
            return_tensors="pt",
        )
        assert result["speech_tensors"].dim() == 3, (
            f"speech_tensors should be 3-D, got {result['speech_tensors'].dim()}-D"
        )
        assert result["speech_tensors"].shape[1] == 1, (
            "Channel dimension should be 1"
        )

    def test_voice_prompt_2d_tensor(self, processor, sample_audio_2d):
        """2-D tensor voice_prompt should be expanded to [batch, 1, T]."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_2d,
            return_tensors="pt",
        )
        assert result["speech_tensors"].dim() == 3, (
            f"speech_tensors should be 3-D, got {result['speech_tensors'].dim()}-D"
        )
        assert result["speech_tensors"].shape[1] == 1, (
            "Channel dimension should be 1"
        )

    def test_voice_prompt_file_path(self, processor, tmp_wav_file):
        """voice_prompt as file path should load and process the audio."""
        result = processor(
            text="Hello world",
            voice_prompt=tmp_wav_file,
            return_tensors="pt",
        )
        assert "speech_tensors" in result, "File path voice_prompt should produce speech_tensors"
        assert result["speech_tensors"].dim() == 3, (
            "speech_tensors from file should be 3-D"
        )

    def test_voice_prompt_nonexistent_file_raises_error(self, processor):
        """Non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            processor(
                text="Hello world",
                voice_prompt="/nonexistent/audio.wav",
                return_tensors="pt",
            )

    def test_voice_prompt_invalid_type_raises_error(self, processor):
        """Invalid type for voice_prompt should raise ValueError."""
        with pytest.raises(ValueError, match="must be"):
            processor(
                text="Hello world",
                voice_prompt=12345,  # int is not valid
                return_tensors="pt",
            )

    def test_speech_masks_shape_matches_token_count(self, processor, sample_audio_tensor):
        """speech_masks time dimension should match the expected token count."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        n_samples = sample_audio_tensor.shape[-1]
        expected_tokens = max(math.ceil(n_samples / 3200), 1)
        assert result["speech_masks"].shape[1] == expected_tokens, (
            f"speech_masks time dim {result['speech_masks'].shape[1]} "
            f"does not match expected {expected_tokens}"
        )

    def test_speech_masks_all_true(self, processor, sample_audio_tensor):
        """All values in speech_masks should be True (all frames valid)."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        assert result["speech_masks"].all(), "All speech_masks values should be True"

    def test_speech_input_mask_true_count_matches_tokens(self, processor, sample_audio_tensor):
        """Number of True values in speech_input_mask should match voice token count."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        n_samples = sample_audio_tensor.shape[-1]
        expected_tokens = max(math.ceil(n_samples / 3200), 1)
        true_count = result["speech_input_mask"].sum().item()
        assert true_count == expected_tokens, (
            f"True count in speech_input_mask ({true_count}) "
            f"does not match expected token count ({expected_tokens})"
        )


# ---------------------------------------------------------------------------
# Parameter priority tests
# ---------------------------------------------------------------------------


class TestParameterPriority:
    """Tests for voice input priority: voice_cache > voice > voice_prompt."""

    def test_voice_cache_overrides_voice(self, processor, voice_cache_full):
        """When both voice_cache and voice are provided, voice_cache wins."""
        result = processor(
            text="Hello world",
            voice="default",
            voice_cache=voice_cache_full,
            return_tensors="pt",
        )
        assert "voice_cache" in result, "voice_cache should be in result"
        assert result["voice_cache"] is voice_cache_full, (
            "voice_cache should be the explicitly provided dict, not loaded from registry"
        )

    def test_voice_cache_overrides_voice_prompt(self, processor, voice_cache_full, sample_audio_tensor):
        """When both voice_cache and voice_prompt are provided, voice_cache wins."""
        result = processor(
            text="Hello world",
            voice_prompt=sample_audio_tensor,
            voice_cache=voice_cache_full,
            return_tensors="pt",
        )
        assert "voice_cache" in result, "voice_cache should be in result"
        assert "speech_tensors" not in result, (
            "speech_tensors should not be present when voice_cache is given"
        )

    def test_voice_overrides_voice_prompt(self, processor, sample_audio_tensor):
        """When both voice and voice_prompt are provided, voice wins (loaded as voice_cache)."""
        result = processor(
            text="Hello world",
            voice="default",
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        assert "voice_cache" in result, (
            "voice should be loaded as voice_cache, overriding voice_prompt"
        )
        assert "speech_tensors" not in result, (
            "speech_tensors should not be present when voice= is given"
        )

    def test_all_three_provided_voice_cache_wins(self, processor, voice_cache_full, sample_audio_tensor):
        """When all three are provided, voice_cache takes highest priority."""
        result = processor(
            text="Hello world",
            voice="default",
            voice_cache=voice_cache_full,
            voice_prompt=sample_audio_tensor,
            return_tensors="pt",
        )
        assert "voice_cache" in result
        assert result["voice_cache"] is voice_cache_full
        assert "speech_tensors" not in result


# ---------------------------------------------------------------------------
# No voice input tests
# ---------------------------------------------------------------------------


class TestNoVoiceInput:
    """Tests when no voice input is provided (random voice)."""

    def test_no_voice_returns_text_ids(self, processor):
        """Without any voice input, text_ids should still be generated."""
        result = processor(text="Hello world", return_tensors="pt")
        assert "text_ids" in result, "text_ids must always be present"

    def test_no_voice_no_voice_cache(self, processor):
        """Without any voice input, voice_cache should not be in result."""
        result = processor(text="Hello world", return_tensors="pt")
        assert "voice_cache" not in result, (
            "voice_cache should not be present without voice input"
        )

    def test_no_voice_no_speech_tensors(self, processor):
        """Without any voice input, speech_tensors should not be in result."""
        result = processor(text="Hello world", return_tensors="pt")
        assert "speech_tensors" not in result, (
            "speech_tensors should not be present without voice input"
        )

    def test_no_voice_speech_input_mask_all_false(self, processor):
        """Without voice input, speech_input_mask should be all False."""
        result = processor(text="Hello world", return_tensors="pt")
        mask = result["speech_input_mask"]
        assert not mask.any(), (
            "speech_input_mask should be all False when no voice input is given"
        )


# ---------------------------------------------------------------------------
# Text handling tests
# ---------------------------------------------------------------------------


class TestTextHandling:
    """Tests for text processing in the processor."""

    def test_text_required(self, processor):
        """Calling processor without text should raise ValueError."""
        with pytest.raises(ValueError, match="[Tt]ext"):
            processor(return_tensors="pt")

    def test_speaker_prefix_auto_added(self, processor):
        """Text without 'Speaker' prefix should get 'Speaker 0:' prepended."""
        # We can test this indirectly by checking the tokenizer was called with formatted text
        result = processor(text="Hello world", return_tensors="pt")
        # The tokenizer mock records calls; check that "Speaker 0:" appears
        calls = processor.tokenizer.encode.call_args_list
        found_speaker = any("Speaker 0:" in str(call) for call in calls)
        assert found_speaker, "Processor should auto-add 'Speaker 0:' prefix"


# ---------------------------------------------------------------------------
# return_tensors tests
# ---------------------------------------------------------------------------


class TestReturnTensors:
    """Tests for return_tensors parameter behavior."""

    def test_return_tensors_pt(self, processor):
        """return_tensors='pt' should produce torch.Tensor for text_ids."""
        result = processor(text="Hello world", return_tensors="pt")
        assert isinstance(result["text_ids"], torch.Tensor), (
            "text_ids should be a tensor with return_tensors='pt'"
        )

    def test_return_tensors_none(self, processor):
        """return_tensors=None should produce a plain list for text_ids."""
        result = processor(text="Hello world", return_tensors=None)
        assert isinstance(result["text_ids"], list), (
            "text_ids should be a list with return_tensors=None"
        )
