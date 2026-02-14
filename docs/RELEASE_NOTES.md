# Release Notes: Voice Cloning Restoration

**Date:** 2026-02-13
**Version:** 0.1.0

---

## Overview

This release restores full voice cloning functionality to KugelAudio Open Source. Voice cloning was previously removed to simplify the initial open-source release (`ce0ce23`). It has now been rebuilt across 12 implementation tasks, adding the ability to clone any speaker's voice from a short reference audio clip or use pre-encoded voice files.

All changes are backward-compatible. Existing code that uses KugelAudio without voice cloning continues to work without modification.

---

## Features Restored

### Voice Cloning Capabilities

- **On-the-fly voice cloning** from raw audio files (WAV, MP3, FLAC) via the `voice_prompt` parameter.
- **Named pre-encoded voices** from a voice registry (`voices/voices.json`) via the `voice` parameter.
- **Pre-encoded voice cache** for production workloads via the `voice_cache` parameter.

### Three Voice Input Methods

| Priority | Parameter | Description | Encoders Required? |
|----------|-----------|-------------|-------------------|
| 1 (highest) | `voice_cache` | Pre-encoded voice features dict | No |
| 2 | `voice` | Named voice from the registry | No |
| 3 (lowest) | `voice_prompt` | Raw audio file or tensor | Yes |

When multiple methods are provided, the highest-priority method wins.

### Dual-Encoder Architecture

The model uses two complementary encoders to capture speaker identity:

- **Acoustic encoder** -- Extracts low-level audio features (timbre, pitch, tone quality) and produces `acoustic_mean` / `acoustic_std` tensors.
- **Semantic encoder** -- Captures higher-level speech patterns (prosody, rhythm, speaking style) and produces a `semantic_mean` tensor.

Both encoder outputs are projected through connector layers and combined to form the voice embedding used during generation.

### Multilingual Support

Voice cloning works across all 23 supported European languages. A voice encoded from one language can be used to generate speech in another, preserving the speaker's timbre while adapting to the target language.

### Default Voices

Three pre-encoded voices ship with the model:

| Voice | Description | File |
|-------|-------------|------|
| `default` | Default neutral voice | `voices/default.pt` |
| `warm` | Warm, friendly voice | `voices/warm.pt` |
| `clear` | Clear, professional voice | `voices/clear.pt` |

---

## What's New

### API Additions

- **`model.encode_voice_prompt(voice_audio, sample_rate=24000)`** -- Encode a reference audio clip into a reusable voice cache dict. Returns a dict with `acoustic_mean`, `acoustic_std`, `semantic_mean`, `audio_length`, and `sample_rate`.

- **`voice_prompt` parameter** on `KugelAudioProcessor.__call__()` -- Pass a file path or raw audio tensor to clone a voice on the fly. The processor handles loading, resampling, and packaging the audio for the model.

- **`voice` parameter** on `KugelAudioProcessor.__call__()` -- Pass the name of a registered voice (e.g., `"default"`, `"warm"`, `"clear"`). The processor loads the corresponding `.pt` file from the voice registry.

- **`voice_cache` parameter** on `KugelAudioProcessor.__call__()` -- Pass a pre-encoded voice dict directly. Highest priority among the three voice input methods.

- **`processor.get_available_voices()`** -- Returns a list of registered voice names from `voices/voices.json`.

- **`processor.load_voice_cache(voice_name)`** -- Load a pre-encoded voice by name from the registry.

- **`model.model.strip_encoders()`** -- Remove encoder weights to free VRAM. Only use when exclusively relying on pre-encoded voices.

- **`KugelAudioSemanticTokenizerConfig`** -- Configuration class for the semantic tokenizer, now exported from the package.

- **`KugelAudioSemanticTokenizerModel`** -- Model class for the semantic tokenizer, now exported from the package.

### Utilities

- **`scripts/create_voice.py`** -- Command-line utility to encode an audio file into a reusable `.pt` voice file with metadata. Supports `--input`, `--output`, `--name`, `--description`, `--language`, and `--model` flags.

### Documentation

- **`docs/VOICE_CLONING.md`** -- Comprehensive voice cloning guide covering all three input methods, audio quality guidelines, custom voice creation, VRAM considerations, multilingual support, advanced usage patterns, troubleshooting, responsible use, and full API reference.

- **`examples/voice_cloning.py`** -- Complete example script demonstrating all three voice input methods with inline documentation.

- **`examples/basic_generation.py`** -- Updated to reference voice cloning capabilities and show the `voice` parameter.

- **`README.md`** -- Updated with a Voices section, Python API examples for all three methods, and links to the voice cloning guide.

### Tests

- **92 tests passing, 3 skipped** (real-weights tests that require model download).
- **`tests/test_voice_encoding.py`** -- Tests for `encode_voice_prompt()`: return structure, tensor shapes, input handling (1D/2D/3D tensors, file paths), error cases (stripped encoders, nonexistent files), and CPU output.
- **`tests/test_processor.py`** -- Tests for processor voice handling: `get_available_voices()`, `load_voice_cache()`, `voice` parameter, `voice_cache` parameter, `voice_prompt` parameter, parameter priority, no-voice fallback, text handling, and return tensor formats.
- **`tests/test_generation.py`** -- Tests for generation with voice inputs: input validation, output structure, voice prompt legacy path, `_process_speech_inputs()` modes, and `load_voice()`.
- **`tests/test_integration.py`** -- Integration tests: config, model architecture, backward compatibility, voice reuse, processor-model integration, stripped encoders, and multilingual voices.

---

## Migration Guide

### No Breaking Changes

This release introduces no breaking changes. All existing APIs continue to work as before.

- Code that does not use voice parameters will produce identical results.
- The `text_ids` / `input_ids` parameter on `generate()` works exactly as before.
- The processor's text handling, return format, and tokenization are unchanged.

### Optional Migration Steps

If you want to adopt voice cloning in existing code:

1. **Simplest approach** -- Add `voice="default"` to your processor call:
   ```python
   # Before
   inputs = processor(text="Hello!", return_tensors="pt")

   # After (uses default voice)
   inputs = processor(text="Hello!", voice="default", return_tensors="pt")
   ```

2. **Clone a specific voice** -- Add `voice_prompt="speaker.wav"`:
   ```python
   inputs = processor(text="Hello!", voice_prompt="speaker.wav", return_tensors="pt")
   ```
   Note: Do not call `strip_encoders()` if using `voice_prompt`.

3. **Production deployment** -- Encode voices once, strip encoders, and use `voice_cache`:
   ```python
   voice_cache = model.encode_voice_prompt("speaker.wav")
   torch.save(voice_cache, "my_voice.pt")
   model.model.strip_encoders()  # Free VRAM

   voice_cache = torch.load("my_voice.pt", map_location="cpu", weights_only=True)
   inputs = processor(text="Hello!", voice_cache=voice_cache, return_tensors="pt")
   ```

---

## Technical Details

### Dual-Encoder Architecture

The voice cloning pipeline processes reference audio through two parallel encoders:

```
Reference Audio
    |
    +---> Acoustic Encoder ---> acoustic_mean / acoustic_std
    |                                  |
    |                         Acoustic Connector
    |                                  |
    +---> Semantic Encoder ---> semantic_mean
                                       |
                              Semantic Connector
                                       |
                              Combined Embedding  --->  TTS Generation
```

- The acoustic encoder captures timbre, pitch, and tonal characteristics.
- The semantic encoder captures prosody, rhythm, and speaking style.
- Both outputs are projected through learned connector layers and summed.
- The combined embedding is injected into the generation pipeline.

### Voice Input Priority

When multiple voice parameters are provided, the processor follows a strict priority order:

1. `voice_cache` (highest) -- Direct pre-encoded dict.
2. `voice` -- Named voice resolved from `voices/voices.json`.
3. `voice_prompt` (lowest) -- Raw audio requiring on-the-fly encoding.

This design ensures deterministic behavior and allows users to override lower-priority defaults.

### VRAM Considerations

- The dual encoders (acoustic + semantic) consume a meaningful portion of VRAM.
- If using only pre-encoded voices (`voice` or `voice_cache`), call `model.model.strip_encoders()` to free this memory.
- After stripping, `voice_prompt` and `encode_voice_prompt()` will raise `RuntimeError`.

---

## Files Changed

### New Files

| File | Description |
|------|-------------|
| `voices/default.pt` | Default neutral voice embedding |
| `voices/warm.pt` | Warm, friendly voice embedding |
| `voices/clear.pt` | Clear, professional voice embedding |
| `voices/samples/default.wav` | Source audio for default voice |
| `voices/samples/warm.wav` | Source audio for warm voice |
| `voices/samples/clear.wav` | Source audio for clear voice |
| `scripts/create_voice.py` | Voice creation utility |
| `examples/voice_cloning.py` | Voice cloning example script |
| `docs/VOICE_CLONING.md` | Comprehensive voice cloning guide |
| `docs/RELEASE_NOTES.md` | This file |
| `tests/test_voice_encoding.py` | Voice encoding tests |
| `tests/test_processor.py` | Processor voice handling tests |
| `tests/test_generation.py` | Generation with voice input tests |
| `tests/test_integration.py` | Integration tests |
| `tests/conftest.py` | Shared test fixtures |
| `tests/_helpers.py` | Test helper utilities |

### Modified Files

| File | Changes |
|------|---------|
| `src/kugelaudio_open/__init__.py` | Added voice cloning classes and docstring to exports |
| `src/kugelaudio_open/configs/` | Restored `KugelAudioSemanticTokenizerConfig` |
| `src/kugelaudio_open/models/` | Restored dual-encoder architecture, `encode_voice_prompt()`, voice prompt support in `generate()` |
| `src/kugelaudio_open/processors/` | Added `voice`, `voice_cache`, `voice_prompt` parameters; `get_available_voices()`; `load_voice_cache()` |
| `examples/basic_generation.py` | Updated with voice cloning references |
| `voices/voices.json` | Updated voice registry with entries for all three voices |
| `README.md` | Added Voices section, updated Python API examples, linked to voice cloning guide |

### Commit History

```
5de8d1e docs: fix terminology and language count consistency
118bc65 docs: add comprehensive voice cloning documentation
0475244 feat: update package exports for voice cloning
cbd0070 test: fix P1 issues (DRY violations and weak assertions)
59ec92c test: add comprehensive voice cloning tests
0dd30db fix: address documentation P1 issues (dtype and strip_encoders notes)
64171fe docs: update examples and README with voice cloning documentation
b01ccef fix: add trailing newlines to voices.json and .gitignore
b4ddc54 feat: add default voice files (default, warm, clear)
30acd83 feat: add voice creation utility script
c7fa147 feat: update generate() to support voice_prompt
c1d62bb feat: add voice_prompt parameter to processor
f71ec95 feat: restore dual-encoder processing in _process_speech_inputs
efac2f5 feat: restore encode_voice_prompt() method
a97b844 feat: restore semantic tokenizer and connector in model
c8e3cb4 feat: restore semantic_tokenizer_config to model config
```

---

## Verification Checklist

- [x] All tests pass (92 passed, 3 skipped, 0 failures)
- [x] All example scripts are syntactically valid (`basic_generation.py`, `voice_cloning.py`)
- [x] All example imports resolve correctly
- [x] All three default voice files exist and load (`default.pt`, `warm.pt`, `clear.pt`)
- [x] Voice files contain expected keys (`acoustic_mean`, `acoustic_std`, `semantic_mean`, etc.)
- [x] Voice registry (`voices.json`) is valid JSON with entries for all three voices
- [x] `create_voice.py` script is executable and documented
- [x] `VOICE_CLONING.md` is comprehensive (648 lines, covers all topics)
- [x] `README.md` is updated with voice cloning info and links
- [x] Package exports include all voice cloning classes (`__all__` has 17 entries)
- [x] No breaking changes introduced (all existing APIs unchanged)
- [x] MIT license compliance maintained (`LICENSE` file present and valid)
- [x] All voice files are tracked in git
- [x] All source audio samples are tracked in git

---

## License

MIT License -- Copyright (c) 2026 Kajo Kratzenstein. See [LICENSE](../LICENSE) for details.
