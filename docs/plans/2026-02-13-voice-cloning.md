# Voice Cloning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore full voice cloning capabilities with acoustic + semantic encoders, supporting flexible voice inputs (pre-encoded, on-the-fly, cached) while maintaining backward compatibility.

**Architecture:** Reverse commit ce0ce23 to restore semantic tokenizer and encode_voice_prompt() method. Enhance processor to handle voice_prompt parameter for audio file input. Create utility scripts for voice file generation and ship default voices.

**Tech Stack:** PyTorch, transformers, librosa (audio processing), soundfile (audio I/O)

---

## Summary of Tasks

This plan contains 13 tasks broken down into bite-sized steps (2-5 minutes each):

1. Restore semantic tokenizer configuration
2. Restore semantic tokenizer in model architecture
3. Restore encode_voice_prompt() method
4. Restore _process_speech_inputs() for both encoders
5. Add voice_prompt parameter to processor
6. Update generate() method for voice_prompt
7. Create voice generation utility script
8. Create default voice files (default.pt, warm.pt, clear.pt)
9. Update examples and documentation
10. Final integration testing
11. Update package exports
12. Final documentation and README updates
13. Final verification and release prep

See the full plan with detailed TDD steps for each task in this file.

## Success Criteria

- Users can clone voices from audio files
- Pre-encoded voices work out of the box
- Backward compatibility maintained
- Voice quality matches original VibeVoice
- Multilingual voice cloning works
- Clean, MIT-licensed code
- Comprehensive tests and documentation

---

For the complete detailed implementation plan with TDD steps, tests, and exact file modifications, see the full plan document.
