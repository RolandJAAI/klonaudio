# Voice Cloning Guide

Comprehensive guide to voice cloning with KugelAudio, covering all input methods, best practices, and production deployment.

---

**Table of Contents**

- [Overview](#overview)
- [Voice Input Methods](#voice-input-methods)
- [Quick Start Examples](#quick-start-examples)
- [Audio Quality Guidelines](#audio-quality-guidelines)
- [Creating Custom Voices](#creating-custom-voices)
- [VRAM Considerations](#vram-considerations)
- [Multilingual Support](#multilingual-support)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Responsible Use](#responsible-use)
- [API Reference](#api-reference)

---

## Overview

### What Is Voice Cloning?

Voice cloning allows you to generate speech that sounds like a specific speaker. Given a short reference audio clip (5--10 seconds), KugelAudio extracts the speaker's vocal characteristics and uses them to synthesize new speech in that voice.

### How It Works

KugelAudio uses a **dual-encoder architecture** to capture two complementary aspects of a speaker's voice:

1. **Acoustic encoder** -- Extracts low-level audio features (timbre, pitch, tone quality) and produces a latent distribution (`acoustic_mean`, `acoustic_std`).
2. **Semantic encoder** -- Captures higher-level speech patterns (prosody, rhythm, speaking style) and produces a latent representation (`semantic_mean`).

At inference time, the outputs of both encoders are projected through connector layers and summed to form a combined voice embedding. This embedding is injected into the generation pipeline so that the synthesized speech faithfully reproduces the reference speaker.

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

### Use Cases

- **Content creation** -- Maintain a consistent narrator voice across episodes of a podcast or audiobook.
- **Accessibility** -- Generate personalized speech for users who have lost the ability to speak.
- **Localization** -- Keep the same brand voice when generating speech in different languages.
- **Prototyping** -- Quickly test how text sounds in a particular voice before recording with a real speaker.

---

## Voice Input Methods

KugelAudio supports three ways to provide a speaker voice. They are listed below in **priority order** -- if multiple are supplied, the highest-priority method wins.

| Priority | Parameter | Description | Encoders Required? |
|----------|-----------|-------------|-------------------|
| 1 (highest) | `voice_cache` | Pre-encoded voice features dict | No |
| 2 | `voice` | Named voice from the registry (`voices.json`) | No |
| 3 (lowest) | `voice_prompt` | Raw audio file or tensor | **Yes** |

### `voice_prompt` -- Clone from audio on the fly

Pass a file path (or raw tensor) to a reference audio clip. The model encodes it through both encoders at inference time.

**When to use:** Quick experiments, one-off cloning, testing new voices.

**Trade-off:** Requires the encoder weights to be loaded, which uses more VRAM. Re-encodes the audio on every call.

### `voice` -- Named pre-encoded voice

Pass the name of a voice registered in `voices/voices.json`. The processor loads the corresponding `.pt` file automatically.

**When to use:** Using built-in voices (`default`, `warm`, `clear`) or custom voices you have already created.

**Trade-off:** Limited to voices that have been pre-encoded and registered.

### `voice_cache` -- Pre-encoded dict

Pass a Python dict containing the pre-encoded tensors (`acoustic_mean`, `semantic_mean`, etc.) directly.

**When to use:** Production deployments where you encode once and serve many requests, or when you need fine-grained control over the voice representation.

**Trade-off:** Requires an initial encoding step, but after that the encoders can be stripped to save VRAM.

---

## Quick Start Examples

### Method 1: Clone from an audio file

```python
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "Roland-JAAI/klonaudio", torch_dtype=dtype,
).to(device)
model.eval()

# NOTE: Do NOT call strip_encoders() -- encoders are needed for voice_prompt.

processor = KugelAudioProcessor.from_pretrained("Roland-JAAI/klonaudio")

inputs = processor(
    text="Hello, this is a cloned voice.",
    voice_prompt="path/to/reference_audio.wav",
    return_tensors="pt",
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "cloned.wav")
```

### Method 2: Use a named pre-encoded voice

```python
# Same model/processor setup as above.
# You CAN call model.model.strip_encoders() here to save VRAM.

print(processor.get_available_voices())  # ["default", "warm", "clear"]

inputs = processor(
    text="Hello from the warm voice.",
    voice="warm",
    return_tensors="pt",
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "warm_output.wav")
```

### Method 3: Pre-encode and cache a voice

```python
# Step 1: Encode once (encoders must be loaded)
voice_cache = model.encode_voice_prompt("path/to/reference_audio.wav")
torch.save(voice_cache, "my_voice.pt")

# Step 2: Strip encoders to save VRAM (optional, but recommended for production)
model.model.strip_encoders()

# Step 3: Reuse the cached voice
voice_cache = torch.load("my_voice.pt", map_location="cpu", weights_only=True)

inputs = processor(
    text="Reusing the pre-encoded voice.",
    voice_cache=voice_cache,
    return_tensors="pt",
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "cached_voice_output.wav")
```

For a complete, runnable script covering all three methods, see [`examples/voice_cloning.py`](../examples/voice_cloning.py).

---

## Audio Quality Guidelines

The quality of the reference audio has a significant impact on the cloned voice. Follow these guidelines for best results.

### Recommended Sample Length

- **Ideal:** 5--10 seconds of speech.
- **Minimum:** 3 seconds (shorter clips may not capture enough of the speaker's characteristics).
- **Maximum:** There is no hard upper limit, but longer clips increase encoding time and VRAM usage without proportional quality gains.

### Sample Rate

- **Target:** 24,000 Hz (24 kHz). The model's audio processor operates at this rate.
- Audio at other sample rates is automatically resampled, but starting at 24 kHz avoids any resampling artifacts.

### Audio Quality Tips

| Do | Avoid |
|----|-------|
| Use a clean, dry recording | Background music or noise |
| Record in a quiet room | Reverb or echo |
| Use a close microphone | Distant or ambient microphones |
| Capture natural, steady speech | Whispering, shouting, or singing (unless that style is desired) |
| Ensure consistent volume | Clipping or very quiet audio |

### Mono vs Stereo

- The encoder expects **mono** audio. Stereo files are accepted but will be mixed down to mono internally.
- For best control, provide mono files directly.

### File Formats

The audio processor supports common formats including WAV, MP3, and FLAC. Uncompressed WAV at 24 kHz mono is recommended.

---

## Creating Custom Voices

### Using `create_voice.py`

The `scripts/create_voice.py` utility encodes a reference audio file into a reusable `.pt` voice file.

```bash
python scripts/create_voice.py \
    --input speaker.wav \
    --output voices/my_voice.pt \
    --name "My Voice" \
    --description "Warm male narrator" \
    --language en
```

**Options:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--input` | `-i` | Path to the input audio file (WAV, MP3, FLAC, etc.) | *required* |
| `--output` | `-o` | Path for the output `.pt` voice file | *required* |
| `--model` | `-m` | Model ID or local path to load | `Roland-JAAI/klonaudio` |
| `--name` | `-n` | Human-readable name for this voice | Derived from filename |
| `--description` | `-d` | Description of voice characteristics | `""` |
| `--language` | `-l` | Language code for the voice | `en` |

The utility automatically:
1. Loads the model with encoders.
2. Encodes the audio through both acoustic and semantic encoders.
3. Attaches metadata (name, description, language, source file, creation timestamp).
4. Saves the result as a `.pt` file.
5. Verifies the saved file can be loaded correctly.

### Recording Reference Audio

For best results when recording your own reference audio:

1. Use a quality microphone in a quiet environment.
2. Record 5--10 seconds of natural, conversational speech.
3. Export as WAV at 24 kHz, 16-bit, mono.
4. Trim silence from the beginning and end.
5. Normalize volume to around -25 dB FS (the processor does this automatically, but starting close helps).

### Voice Registry (`voices.json`)

Pre-encoded voices are registered in `voices/voices.json`. Each entry maps a voice name to its file and metadata:

```json
{
  "my_voice": {
    "file": "my_voice.pt",
    "description": "Warm male narrator",
    "language": "en",
    "gender": "neutral",
    "source": "recorded",
    "encoded_with": "kugelaudio-0-open"
  }
}
```

After adding an entry, the voice becomes available through the `voice` parameter:

```python
inputs = processor(text="Hello!", voice="my_voice", return_tensors="pt")
```

### Built-in Voices

KugelAudio ships with three pre-encoded voices:

| Voice | Description |
|-------|-------------|
| `default` | Default neutral voice |
| `warm` | Warm, friendly voice |
| `clear` | Clear, professional voice |

These are stored as `.pt` files in the `voices/` directory of the model repository and are downloaded automatically from HuggingFace when first used.

---

## VRAM Considerations

### Encoder Memory Usage

The dual voice encoders (acoustic + semantic) consume a meaningful portion of VRAM. If you only use pre-encoded voices (`voice` or `voice_cache`), you can free this memory.

### Stripping Encoders

Call `strip_encoders()` to remove the encoder weights:

```python
model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "Roland-JAAI/klonaudio", torch_dtype=torch.bfloat16,
).to("cuda")
model.eval()

# Free encoder VRAM -- only do this if you will NOT use voice_prompt
model.model.strip_encoders()
```

After stripping:
- `voice` and `voice_cache` continue to work normally.
- `voice_prompt` will raise a `RuntimeError`.
- `encode_voice_prompt()` will raise a `RuntimeError`.

### Trade-offs Summary

| Method | Encoders Needed | VRAM After Stripping | Latency |
|--------|----------------|---------------------|---------|
| `voice_prompt` | Yes | N/A (cannot strip) | Higher (encoding per request) |
| `voice` / `voice_cache` | No | Lower | Lower (no encoding step) |

**Recommendation:** For production, pre-encode all voices with `encode_voice_prompt()` or `create_voice.py`, then call `strip_encoders()` to minimize VRAM.

---

## Multilingual Support

KugelAudio supports **23 European languages**. Voice cloning works across all supported languages.

### Supported Languages

| Language | Code | Language | Code | Language | Code |
|----------|------|----------|------|----------|------|
| English | `en` | German | `de` | French | `fr` |
| Spanish | `es` | Italian | `it` | Portuguese | `pt` |
| Dutch | `nl` | Polish | `pl` | Russian | `ru` |
| Ukrainian | `uk` | Czech | `cs` | Romanian | `ro` |
| Hungarian | `hu` | Swedish | `sv` | Danish | `da` |
| Finnish | `fi` | Norwegian | `no` | Greek | `el` |
| Bulgarian | `bg` | Slovak | `sk` | Croatian | `hr` |
| Serbian | `sr` | Turkish | `tr` | | |

### Cross-Language Voice Cloning

You can clone a voice from one language and generate speech in another. For example, encode a German speaker's voice and use it to generate English speech. The model preserves the speaker's timbre and tone quality while adapting to the target language's phonetics.

**Notes:**
- Quality varies by language. Spanish, French, English, and German have the strongest representation in the training data.
- Some languages with less training data may show reduced quality in prosody and pronunciation.
- The voice characteristics (timbre, pitch) transfer well across languages; language-specific prosody may differ from the reference.

---

## Advanced Usage

### Voice Caching for Production

In a production environment, avoid re-encoding voices on every request. Instead, pre-encode once and serve from cache:

```python
import torch
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)

model_id = "Roland-JAAI/klonaudio"
device = "cuda"
dtype = torch.bfloat16

# --- Startup: encode voices ---
model = KugelAudioForConditionalGenerationInference.from_pretrained(
    model_id, torch_dtype=dtype,
).to(device)
model.eval()

# Encode all custom voices
voices = {}
for name, path in [("alice", "alice.wav"), ("bob", "bob.wav")]:
    voices[name] = model.encode_voice_prompt(path)
    torch.save(voices[name], f"voices/{name}.pt")

# Free encoder VRAM
model.model.strip_encoders()

processor = KugelAudioProcessor.from_pretrained(model_id)

# --- Serving: generate with cached voices ---
def generate(text: str, voice_name: str) -> torch.Tensor:
    inputs = processor(
        text=text,
        voice_cache=voices[voice_name],
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0)
    return outputs.speech_outputs[0]
```

### Batch Processing

Generate speech for multiple texts with the same voice:

```python
texts = [
    "First sentence to synthesize.",
    "Second sentence to synthesize.",
    "Third sentence to synthesize.",
]

voice_cache = torch.load("my_voice.pt", map_location="cpu", weights_only=True)

for i, text in enumerate(texts):
    inputs = processor(text=text, voice_cache=voice_cache, return_tensors="pt")
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0)
    processor.save_audio(outputs.speech_outputs[0], f"output_{i}.wav")
```

### Voice Reuse Across Sessions

Save a voice once and load it in any session, on any machine:

```python
# Session 1: Create voice
voice_cache = model.encode_voice_prompt("speaker.wav")
torch.save(voice_cache, "my_voice.pt")

# Session 2 (possibly different machine): Load and use
voice_cache = torch.load("my_voice.pt", map_location="cpu", weights_only=True)
inputs = processor(text="Hello!", voice_cache=voice_cache, return_tensors="pt")
```

The `.pt` files are portable and self-contained. They include metadata (name, description, language, source file, creation timestamp, model ID) alongside the encoded tensors.

---

## Troubleshooting

### Stripped Encoder Error

**Error:**
```
RuntimeError: Cannot encode voice prompt: acoustic tokenizer encoder has been stripped.
Do not call strip_encoders() if you need voice cloning.
```

**Cause:** You called `model.model.strip_encoders()` but then tried to use `voice_prompt` or `encode_voice_prompt()`.

**Fix:** Remove the `strip_encoders()` call, or switch to `voice` / `voice_cache` which do not require encoders.

### File Not Found

**Error:**
```
FileNotFoundError: Voice prompt file not found: path/to/audio.wav
```

**Cause:** The file path passed to `voice_prompt` does not exist.

**Fix:** Verify the file path is correct and the file exists. Use an absolute path if in doubt.

### Voice Not Found in Registry

**Error:**
```
ValueError: Voice 'my_voice' not found. Available voices: default, warm, clear
```

**Cause:** The voice name passed to the `voice` parameter is not registered in `voices/voices.json`.

**Fix:** Check available voices with `processor.get_available_voices()`. Either use an available name or add your voice to the registry.

### Poor Voice Quality

If the cloned voice does not sound like the reference speaker:

1. **Check reference audio quality.** Clean, close-mic recordings produce the best clones. Background noise, reverb, and music degrade quality significantly.
2. **Check audio length.** Very short clips (under 3 seconds) may not capture enough speaker information. Aim for 5--10 seconds.
3. **Try a different reference clip.** Some recordings simply work better than others. Choose a clip with natural, steady speech.
4. **Adjust `cfg_scale`.** Higher values (e.g., 5.0) increase adherence to the text at the cost of naturalness. Lower values (e.g., 1.5) may produce more natural but less precise output. The default of 3.0 is a good starting point.

### Missing `acoustic_mean` in Voice File

**Error:**
```
ValueError: Voice file 'my_voice.pt' does not contain 'acoustic_mean'.
```

**Cause:** The `.pt` file was not created by the KugelAudio voice encoding pipeline, or it was corrupted.

**Fix:** Re-create the voice file using `encode_voice_prompt()` or `scripts/create_voice.py`.

---

## Responsible Use

### Consent

**Always obtain explicit consent** before cloning someone's voice. Voice cloning technology should be used to empower speakers, not to impersonate them without permission.

### Ethical Considerations

- Do not create content that could be mistaken for a real person speaking without their knowledge.
- Do not use voice cloning for fraud, deception, or manipulation.
- Be transparent with audiences when synthesized speech is used in published content.

### Deepfake Concerns

KugelAudio automatically watermarks all generated audio using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal). This imperceptible watermark can be detected programmatically to verify whether audio was AI-generated:

```python
from kugelaudio_open.watermark import AudioWatermark

watermark = AudioWatermark()
result = watermark.detect(audio, sample_rate=24000)
print(f"AI-generated: {result.detected}, confidence: {result.confidence:.1%}")
```

### License

KugelAudio is released under the MIT License. You are free to use it for personal and commercial purposes. However, you are responsible for ensuring your use complies with applicable laws and ethical standards.

---

## API Reference

### `model.encode_voice_prompt(voice_audio, sample_rate=24000)`

Encode a reference audio clip into a reusable voice cache dict.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `voice_audio` | `str` or `torch.Tensor` | File path to a WAV/MP3/FLAC file, or a raw audio tensor with shape `[batch, channels, samples]` |
| `sample_rate` | `int` | Sample rate of the input audio. Default: `24000` |

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `acoustic_mean` | `torch.Tensor` | Acoustic encoder output mean |
| `acoustic_std` | `torch.Tensor` | Acoustic encoder output standard deviation |
| `semantic_mean` | `torch.Tensor` | Semantic encoder output mean |
| `audio_length` | `int` | Original audio length in samples |
| `sample_rate` | `int` | Sample rate used |

**Raises:** `RuntimeError` if encoders have been stripped.

---

### `processor(text, voice=None, voice_cache=None, voice_prompt=None, ...)`

The `KugelAudioProcessor.__call__` method. Prepares text and voice inputs for the model.

**Voice parameters (in priority order):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `voice_cache` | `dict` or `None` | Pre-encoded voice features dict (highest priority) |
| `voice` | `str` or `None` | Name of a registered voice from `voices.json` |
| `voice_prompt` | `str`, `torch.Tensor`, or `None` | Path to audio file or raw audio tensor (lowest priority) |

**Other parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text to synthesize (required) |
| `return_tensors` | `str` or `None` | Return format, typically `"pt"` |

**Returns:** `BatchEncoding` containing `text_ids`, `speech_input_mask`, and either `voice_cache` or `speech_tensors`/`speech_masks` depending on the voice input method.

---

### `processor.get_available_voices()`

Returns a list of registered voice names.

```python
processor.get_available_voices()  # ["default", "warm", "clear"]
```

---

### `processor.load_voice_cache(voice_name)`

Load a pre-encoded voice by name from the registry.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `voice_name` | `str` | Name of the voice (must be in `voices.json`) |

**Returns:** `dict` with `acoustic_mean` and other encoded tensors.

**Raises:** `ValueError` if the voice name is not found in the registry.

---

### `model.model.strip_encoders()`

Remove encoder weights from both tokenizers to free VRAM.

**Warning:** After calling this method, `voice_prompt` and `encode_voice_prompt()` will no longer work. Only use when you exclusively rely on pre-encoded voices.

```python
model.model.strip_encoders()
```

---

### `model.generate(**inputs, cfg_scale=3.0, max_new_tokens=2048)`

Generate speech from processed inputs.

**Key parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_ids` | `torch.Tensor` | *required* | Tokenized text from the processor |
| `speech_input_mask` | `torch.Tensor` | *required* | Mask for voice embedding positions |
| `voice_cache` | `dict` | `None` | Pre-encoded voice features |
| `speech_tensors` | `torch.Tensor` | `None` | Raw audio for on-the-fly encoding |
| `speech_masks` | `torch.Tensor` | `None` | Valid frame mask for `speech_tensors` |
| `cfg_scale` | `float` | `3.0` | Classifier-free guidance scale (1.0--10.0) |
| `max_new_tokens` | `int` | `2048` | Maximum tokens to generate |

**Returns:** `KugelAudioGenerationOutput` with `speech_outputs` list of audio tensors.
