# Voice Cloning Feature Design

**Date:** 2026-02-13
**Author:** KugelAudio Team
**Status:** Approved for Implementation
**License:** MIT

## Executive Summary

This design restores voice cloning capabilities to kugelaudio-open by re-implementing the acoustic and semantic encoder architecture that was removed in commit ce0ce23. The implementation will maintain backward compatibility with the pre-encoded voice system while adding flexible voice cloning from audio samples.

## Motivation

The current kugelaudio-open implementation removed voice cloning to save VRAM, but this eliminates a core feature that users need. Without voice cloning:

- Each generation produces a random voice
- Pre-encoded voices (warm, default, clear) don't work because .pt files are missing
- Users cannot clone custom voices from audio samples
- The system is less useful than the original Microsoft VibeVoice

## Goals

1. **Full Voice Cloning:** Support cloning voices from 3-10 second audio samples
2. **Multilingual Support:** Enable voice cloning for all 24 supported European languages
3. **Flexibility:** Support multiple voice input methods (pre-encoded, on-the-fly, cached)
4. **Quality:** Restore full acoustic + semantic encoding for best quality
5. **Backward Compatibility:** Existing code continues to work
6. **MIT Publication:** Clean, well-documented code ready for public release

## Non-Goals

- VRAM optimization (user chose quality over efficiency)
- Streaming/real-time voice cloning (future enhancement)
- Voice conversion (only cloning from reference audio)
- Multiple voices per generation (single voice per synthesis)

## Success Criteria

- Users can clone voices from audio files
- Pre-encoded voices (warm, default, clear) work out of the box
- Backward compatibility maintained (no breaking changes)
- Voice quality matches or exceeds original VibeVoice
- Multilingual voice cloning works for all 24 languages
- Clean, MIT-licensed code ready for publication
- Comprehensive documentation and examples

## Architecture Overview

### Voice Input Methods

The system supports three voice input methods:

1. **Pre-encoded voices**: `processor(text="...", voice="warm")` - loads from voices.json registry
2. **Direct audio path**: `processor(text="...", voice_prompt="path/to/sample.wav")` - encodes on-the-fly
3. **Pre-encoded cache**: `processor(text="...", voice_cache=cached_dict)` - reuses encoded features

### Component Changes

**Files to Modify:**
- `src/kugelaudio_open/configs/model_config.py` - Restore semantic_tokenizer_config
- `src/kugelaudio_open/models/kugelaudio_model.py` - Restore semantic tokenizer and connector
- `src/kugelaudio_open/models/kugelaudio_inference.py` - Restore encode_voice_prompt() and enhance _process_speech_inputs()
- `src/kugelaudio_open/processors/kugelaudio_processor.py` - Add voice_prompt parameter handling
- `examples/voice_cloning.py` - Restore with updated API
- `scripts/create_voice.py` - New utility for creating custom voices

**Pre-encoded Voices:**
- Create `voices/default.pt`, `voices/warm.pt`, `voices/clear.pt`
- Source from Microsoft VibeVoice samples if MIT-compatible
- Or create original recordings
- Support multilingual cloning (all 24 languages)

## Implementation Phases

### Phase 1: Core Restoration
- Restore semantic tokenizer in model config and architecture
- Restore encode_voice_prompt() method
- Update _process_speech_inputs() to handle both encoders
- Remove or make optional strip_encoders() call

### Phase 2: Processor Enhancement
- Add voice_prompt parameter to processor
- Handle audio file loading and preprocessing
- Add speech_tensors/speech_masks to processor outputs
- Update generate() to accept voice_prompt

### Phase 3: Voice Files & Utilities
- Create/source reference audio samples
- Generate default .pt voice files
- Create voice creation utility script
- Update voices.json with metadata

### Phase 4: Testing & Documentation
- Write comprehensive tests
- Update README with voice cloning examples
- Document multilingual usage
- Add migration guide

## Key Design Principles

- **Backward compatible**: No breaking changes
- **Clean separation**: Voice encoding in model, I/O in processor
- **MIT-ready**: Well-documented, clear errors, proper types
- **Multilingual**: Support all 24 languages out of the box
