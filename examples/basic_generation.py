#!/usr/bin/env python3
"""Basic example of generating speech with KugelAudio.

All generated audio is automatically watermarked for identification.

Voice input methods (see examples/voice_cloning.py for full details):
  - voice="angry"           Named pre-encoded voice from voices.json
  - voice_prompt="ref.wav"  Clone a voice from a raw audio file
  - voice_cache={...}       Supply a pre-encoded voice dict directly
"""

import torch

from kugelaudio_open import (
    AudioWatermark,
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)


def main():
    # Configuration
    model_id = "kugelaudio/kugelaudio-0-open"
    # Use MPS on Apple Silicon, CUDA on NVIDIA, or CPU as fallback
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't support bfloat16 well
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device} with dtype: {dtype}")

    print(f"Loading model {model_id}...")

    # Load model and processor
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Strip encoder weights to save VRAM (only decoders needed for inference).
    # NOTE: If you plan to use voice_prompt for voice cloning, do NOT call
    # strip_encoders() -- the encoders are required to process raw audio.
    model.model.strip_encoders()

    processor = KugelAudioProcessor.from_pretrained(model_id)

    # Show available pre-encoded voices
    voices = processor.get_available_voices()
    print(f"Available voices: {voices}")

    # Text to synthesize with the professional radio voice
    text = "Guten Abend und herzlich willkommen zu unserer Sendung. Dies ist eine Demonstration der natürlichen Text-zu-Sprache-Synthese von KugelAudio mit Stimmklonung."

    print(f"Generating speech for: '{text}'")

    # Process input with a named voice (using 'radio' as the default professional voice).
    # Other options:
    #   processor(text=text, voice="angry", ...)            -- use angry voice
    #   processor(text=text, voice="old_lady", ...)         -- use elderly female voice
    #   processor(text=text, voice_prompt="ref.wav", ...)   -- clone from audio (requires removing strip_encoders() above)
    #   processor(text=text, voice_cache=my_cache, ...)     -- pre-encoded dict
    inputs = processor(text=text, voice="radio", return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate speech (watermark is automatically applied)
    # speech_end_penalty=5.0 prevents the model from cutting off the last word
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=3.0,
            max_new_tokens=2048,
            speech_end_penalty=5.0,
        )

    audio = outputs.speech_outputs[0]

    # Verify watermark is present
    print("Verifying watermark...")
    watermark = AudioWatermark()
    result = watermark.detect(audio)
    print(f"Watermark detected: {result.detected}, confidence: {result.confidence:.2%}")

    # Save output
    output_path = "output.wav"
    processor.save_audio(audio, output_path)
    print(f"Audio saved to {output_path}")


if __name__ == "__main__":
    main()
