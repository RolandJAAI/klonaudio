<h1 align="center">🎙️ KlonAudio</h1>

<p align="center">
  <strong>Open-source text-to-speech for European languages with full voice cloning</strong><br>
  Powered by an AR + Diffusion architecture
</p>

<p align="center">
  <a href="https://huggingface.co/kugelaudio/kugelaudio-0-open"><img src="https://img.shields.io/badge/🤗-Hugging_Face_Model-blue" alt="HuggingFace Model"></a>
  <a href="https://github.com/RolandJAAI/klonaudio"><img src="https://img.shields.io/badge/GitHub-Repository-black" alt="GitHub Repository"></a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://huggingface.co/kugelaudio"><img src="https://img.shields.io/badge/🤗-Models-yellow" alt="HuggingFace"></a>
</p>

---

## About KlonAudio

**KlonAudio** is a fork of [KugelAudio](https://github.com/Kugelaudio/kugelaudio-open) with restored voice cloning capabilities. This project re-implements the dual acoustic + semantic encoder architecture that enables high-quality voice cloning from audio samples.

### What's Different

- ✨ **Full Voice Cloning**: Clone any voice from a 5-10 second audio sample
- 🎭 **Dual Encoder System**: Acoustic (voice identity) + Semantic (emotion/prosody) tokenizers restored
- 🎤 **Flexible Voice Input**: Upload files, record live, or use pre-encoded voices
- 🚀 **Gradio UI**: User-friendly web interface with voice cloning built-in

## Motivation

**Open-source text-to-speech models for European languages are significantly lagging behind.** While English TTS has seen remarkable progress, speakers of German, French, Spanish, Polish, and dozens of other European languages have been underserved by the open-source community.

KlonAudio builds on the excellent work by [KugelAudio](https://github.com/Kugelaudio/kugelaudio-open) and the [VibeVoice team at Microsoft](https://github.com/microsoft/VibeVoice), restoring the full voice cloning capabilities using approximately **200,000 hours** of highly pre-processed and enhanced speech data from the [YODAS2 dataset](https://huggingface.co/datasets/espnet/yodas).

## 🏆 Benchmark Results: Outperforming ElevenLabs

**KugelAudio (base model) achieves state-of-the-art performance**, beating industry leaders including ElevenLabs in rigorous human preference testing. KlonAudio inherits this quality and adds full voice cloning capabilities.

### Human Preference Benchmark (A/B Testing)

We conducted extensive A/B testing with **339 human evaluations** to compare KugelAudio against leading TTS models. Participants listened to a reference voice sample, then compared outputs from two models and selected which sounded more human and closer to the original voice.

### German Language Evaluation

The evaluation specifically focused on **German language samples** with diverse emotional expressions and speaking styles:

* **Neutral Speech**: Standard conversational tones
* **Shouting**: High-intensity, elevated volume speech
* **Singing**: Melodic and rhythmic speech patterns
* **Drunken Voice**: Slurred and irregular speech characteristics

These diverse test cases demonstrate the model's capability to handle a wide range of speaking styles beyond standard narration.

### OpenSkill Ranking Results

| Rank | Model | Score | Record | Win Rate |
|------|-------|-------|--------|----------|
| 🥇 1 | **KugelAudio** | **26** | 71W / 20L / 23T | **78.0%** |
| 🥈 2 | ElevenLabs Multi v2 | 25 | 56W / 34L / 22T | 62.2% |
| 🥉 3 | ElevenLabs v3 | 21 | 64W / 34L / 16T | 65.3% |
| 4 | Cartesia | 21 | 55W / 38L / 19T | 59.1% |
| 5 | VibeVoice | 10 | 30W / 74L / 8T | 28.8% |
| 6 | CosyVoice v3 | 9 | 15W / 91L / 8T | 14.2% |

_Based on 339 evaluations using Bayesian skill-rating system (OpenSkill)_

## Audio Samples

Listen to KugelAudio's diverse voice capabilities across different speaking styles and languages:

### German Voice Samples

| Sample | Description | Audio Sample |
|--------|-------------|--------------|
| **Radio Voice** | Professional radio announcer voice | [🔊 Listen](voices/samples/radio_voice.wav) |
| **Angry Voice** | Irritated and frustrated speech | [🔊 Listen](voices/samples/angry.wav) |
| **Old Lady** | Gentle elderly female voice | [🔊 Listen](voices/samples/old_lady.wav) |

*All samples use pre-encoded voice embeddings optimized for German. Click to preview the audio files.*

### Training Details

- **Base Model**: [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)
- **Training Data**: ~200,000 hours from [YODAS2](https://huggingface.co/datasets/espnet/yodas)
- **Hardware**: 8x NVIDIA H100 GPUs
- **Training Duration**: 5 days

### Supported Languages

KugelAudio supports **23 major European languages** with varying levels of quality based on dataset representation:

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|----------|------|------|----------|------|------|----------|------|------|
| English | en | 🇺🇸 | German | de | 🇩🇪 | French | fr | 🇫🇷 |
| Spanish | es | 🇪🇸 | Italian | it | 🇮🇹 | Portuguese | pt | 🇵🇹 |
| Dutch | nl | 🇳🇱 | Polish | pl | 🇵🇱 | Russian | ru | 🇷🇺 |
| Ukrainian | uk | 🇺🇦 | Czech | cs | 🇨🇿 | Romanian | ro | 🇷🇴 |
| Hungarian | hu | 🇭🇺 | Swedish | sv | 🇸🇪 | Danish | da | 🇩🇰 |
| Finnish | fi | 🇫🇮 | Norwegian | no | 🇳🇴 | Greek | el | 🇬🇷 |
| Bulgarian | bg | 🇧🇬 | Slovak | sk | 🇸🇰 | Croatian | hr | 🇭🇷 |
| Serbian | sr | 🇷🇸 | Turkish | tr | 🇹🇷 | | | |

> **📊 Language Coverage Disclaimer**: Quality varies significantly by language. Spanish, French, English, and German have the strongest representation in our training data (~200,000 hours from YODAS2). Other languages may have reduced quality, prosody, or vocabulary coverage depending on their availability in the training dataset.

## 📖 Start Here

Get started with KugelAudio quickly using our documentation:

| | |
|---|---|
| 📥 [**Installation**](#installation) | Set up KlonAudio on your machine |
| 🎯 [**Quick Start**](#quick-start) | Generate your first speech in minutes |
| 🎭 [**Voices**](#voices) | Clone voices or use pre-encoded speakers |
| 📖 [**Voice Cloning Guide**](docs/VOICE_CLONING.md) | In-depth guide to voice cloning |
| 🔒 [**Watermarking**](#audio-watermarking) | Verify AI-generated audio |
| 📦 [**Models**](#models) | Available model variants and benchmarks |

---

## Features

- 🏆 **State-of-the-Art Performance**: Outperforms ElevenLabs and other leading TTS models in human evaluations
- 🌍 **European Language Focus**: Trained specifically for 23 major European languages
- **High-Quality TTS**: State-of-the-art speech synthesis using AR + Diffusion
- 🎭 **Voice Cloning**: Clone any voice from a reference audio file, or use built-in pre-encoded voices
- **Audio Watermarking**: All generated audio is watermarked using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal)
- 🎭 **Emotional Range**: Supports various speaking styles including shouting, singing, and expressive speech
- **Web Interface**: Easy-to-use Gradio UI for non-technical users
- **HuggingFace Integration**: Seamless loading from HuggingFace Hub

## Quick Start

### Installation

#### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (recommended for GPU acceleration)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### Installation

```bash
# Clone the repository
git clone https://github.com/RolandJAAI/klonaudio.git
cd klonaudio

# Run directly with uv (recommended - handles all dependencies automatically)
uv run python start.py
```

That's it! The `uv run` command will automatically create a virtual environment and install all dependencies.

### Launch Web Interface

```bash
# Quick start with uv (recommended)
uv run python start.py

# With a public share link
uv run python start.py ui --share

# Custom host and port
uv run python start.py ui --host 0.0.0.0 --port 8080
```

Then open http://127.0.0.1:7860 in your browser.

### Command Line Usage

```bash
# Generate speech from text
uv run python start.py generate "Hello, this is KugelAudio!" -o hello.wav

# Clone a voice from a reference audio file
uv run python start.py generate "Hello with a cloned voice!" -r speaker.wav -o cloned.wav

# Check if audio contains watermark
uv run python start.py verify audio.wav
```

### Python API

```python
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    torch_dtype=dtype,
).to(device)
model.eval()

processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")

# See available voices
print(processor.get_available_voices())  # ["angry", "radio", "old_lady"]

# Generate speech with a named voice (watermark is automatically applied)
inputs = processor(text="Guten Abend. Hier sind die Nachrichten.", voice="radio", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

# Save audio
processor.save_audio(outputs.speech_outputs[0], "output.wav")

# Or clone a voice from a reference audio file
inputs = processor(text="Hello!", voice_prompt="speaker.wav", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "cloned.wav")
```

> **Tip:** If you only use pre-encoded voices (not `voice_prompt`), call `model.model.strip_encoders()` after loading to save VRAM.

### Voices

> **📖 For a comprehensive guide** including audio quality tips, production caching, multilingual notes, and troubleshooting, see the [Voice Cloning Guide](docs/VOICE_CLONING.md).

KlonAudio supports three ways to control the speaker voice, listed here from simplest to most flexible:

#### 1. Named pre-encoded voice (`voice`)

Select one of the built-in voices by name. The `.pt` files are stored in the model repository and downloaded from HuggingFace automatically.

| Voice | Description | Best For |
|-------|-------------|----------|
| `radio` | Professional radio announcer voice (German) | Default/professional content |
| `angry` | Angry, frustrated voice (German) | Emotional/angry dialogue |
| `old_lady` | Gentle elderly female voice (German) | Storytelling/warm content |

> **Note:** The pre-encoded voices are optimized for German. The model also supports other languages through voice cloning with `voice_prompt`.

```python
voices = processor.get_available_voices()  # ["angry", "radio", "old_lady"]

# Professional radio voice (recommended default)
inputs = processor(text="Guten Abend und herzlich willkommen zur heutigen Sendung.", voice="radio", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "radio_voice_output.wav")
```

#### 2. Voice cloning from audio (`voice_prompt`)

Clone any voice by providing a reference audio file. The model encodes it through its dual acoustic + semantic tokenizers at inference time.

> **Important:** Do **not** call `model.model.strip_encoders()` if you use `voice_prompt` -- the encoders are required to process raw audio.

```python
inputs = processor(
    text="Hello, this clones a voice from an audio file.",
    voice_prompt="path/to/reference_audio.wav",
    return_tensors="pt",
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "cloned_output.wav")
```

#### 3. Pre-encoded voice cache (`voice_cache`)

For production workloads, encode the reference audio once with `encode_voice_prompt()` and reuse the resulting dict. This avoids re-encoding on every request and allows you to call `strip_encoders()` to save VRAM.

```python
# Encode once (requires encoders)
voice_cache = model.encode_voice_prompt("path/to/reference_audio.wav")
torch.save(voice_cache, "my_voice.pt")

# Reuse later (encoders can be stripped)
voice_cache = torch.load("my_voice.pt", map_location="cpu", weights_only=True)
inputs = processor(text="Hello!", voice_cache=voice_cache, return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, cfg_scale=3.0)

processor.save_audio(outputs.speech_outputs[0], "cached_voice_output.wav")
```

#### Creating custom voices from the command line

Use the included utility to create reusable `.pt` voice files:

```bash
python scripts/create_voice.py \
    --input speaker.wav \
    --output voices/my_voice.pt \
    --name "My Voice" \
    --description "Warm male narrator" \
    --language en
```

See `examples/voice_cloning.py` for a complete runnable example of all three methods.

## Audio Watermarking

All audio generated by KugelAudio contains an imperceptible watermark using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal) technology. This helps identify AI-generated content and prevent misuse.

### Verify Watermark

```python
from kugelaudio_open.watermark import AudioWatermark

watermark = AudioWatermark()

# Check if audio is watermarked
result = watermark.detect(audio, sample_rate=24000)

print(f"Detected: {result.detected}")
print(f"Confidence: {result.confidence:.1%}")
```

### Watermark Features

- **Imperceptible**: No audible difference in audio quality
- **Robust**: Survives compression, resampling, and editing
- **Fast Detection**: Real-time capable detection
- **Sample-Level**: 1/16k second resolution

## Models

| Model | Parameters | Quality | RTF | Speed | VRAM |
|-------|------------|---------|-----|-------|------|
| [kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) | 7B | Best | 1.00 | 1.0x realtime | ~19GB |

*RTF = Real-Time Factor (generation time / audio duration). Lower is faster.*
## Architecture

KlonAudio uses a hybrid AR + Diffusion architecture with dual voice encoders:

1. **Text Encoder**: Qwen2-based language model encodes input text
2. **Dual Voice Encoders**:
   - **Acoustic Tokenizer** (64-dim): Captures voice identity and timbre
   - **Semantic Tokenizer** (128-dim): Captures emotion, prosody, and style
3. **TTS Backbone**: Upper transformer layers generate speech representations
4. **Diffusion Head**: Predicts speech latents using denoising diffusion
5. **Acoustic Decoder**: Converts latents to audio waveforms

This dual-encoder design enables both accurate voice cloning and emotional expressiveness.

## Configuration

### Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable TF32 for faster computation on Ampere GPUs
export TORCH_ALLOW_TF32=1
```

### Advanced Generation Parameters

```python
outputs = model.generate(
    **inputs,
    cfg_scale=3.0,                  # Guidance scale (1.0-10.0)
    max_new_tokens=4096,            # Maximum generation length
    speech_end_penalty=5.0,         # Prevents cutting off last word (0.0 to disable)
)
```

## Responsible Use

This technology is intended for legitimate purposes:

✅ **Appropriate Uses:**
- Accessibility (TTS for visually impaired)
- Content creation (podcasts, videos, audiobooks)
- Voice assistants and chatbots
- Language learning applications
- Creative projects with consent

❌ **Prohibited Uses:**
- Creating deepfakes or misleading content
- Impersonating individuals without consent
- Fraud or deception
- Any illegal activities

All generated audio is watermarked to enable detection of AI-generated content.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

KlonAudio would not exist without the outstanding work of:

### Base Model & Architecture

- **[KugelAudio Team](https://github.com/Kugelaudio/kugelaudio-open)** (Kajo Kratzenstein, Carlos Menke): For training the excellent base model and open-sourcing it under MIT license
- **[Microsoft VibeVoice Team](https://github.com/microsoft/VibeVoice)**: For creating the original architecture with dual acoustic + semantic encoders
- **[AI Service Center Berlin-Brandenburg](https://hpi.de/ki-servicezentrum/)**: For providing the GPU resources (8x H100) that made training the base model possible

### Supporting Technologies

- **[YODAS2 Dataset](https://huggingface.co/datasets/espnet/yodas)**: For providing the large-scale multilingual speech data
- **[Qwen Team](https://huggingface.co/Qwen)**: For the powerful language model backbone
- **[Facebook AudioSeal](https://huggingface.co/facebook/audioseal)**: For the audio watermarking technology
- **[HuggingFace](https://huggingface.co)**: For model hosting and the transformers library





## Authors

**Roland Becker**
🏢 [JUST ADD AI GmbH](https://jaai-group.com)

*KlonAudio is a fork that restores voice cloning capabilities to the original KugelAudio project.*

### Original KugelAudio Authors

**Kajo Kratzenstein**
📧 [kajo@kugelaudio.com](mailto:kajo@kugelaudio.com)
🌐 [kugelaudio.com](https://kugelaudio.com)

**Carlos Menke**

## Citation

```bibtex
@software{klonaudio2026,
  title = {KlonAudio: Open-Source Text-to-Speech with Voice Cloning for European Languages},
  author = {Becker, Roland},
  year = {2026},
  organization = {JUST ADD AI GmbH},
  url = {https://github.com/RolandJAAI/klonaudio},
  note = {Fork of KugelAudio with restored voice cloning capabilities}
}

@software{kugelaudio2026,
  title = {KugelAudio: Open-Source Text-to-Speech for European Languages},
  author = {Kratzenstein, Kajo and Menke, Carlos},
  year = {2026},
  institution = {Hasso-Plattner-Institut},
  url = {https://github.com/kugelaudio/kugelaudio}
}
```

---
