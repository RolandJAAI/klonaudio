#!/usr/bin/env python3
"""Voice cloning examples for KugelAudio.

Demonstrates three methods of controlling the speaker voice:
1. voice_prompt  - Clone a voice from a raw audio file (on-the-fly encoding)
2. voice         - Use a named pre-encoded voice from the registry
3. voice_cache   - Supply a pre-encoded voice dict directly

All generated audio is automatically watermarked for identification.

MIT License - Copyright (c) 2026 Kajo Kratzenstein
"""

import torch

from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)


def main():
    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
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

    print(f"Loading model {model_id} ...")
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # NOTE: Do NOT call model.model.strip_encoders() when you need voice
    # cloning.  The encoders are required to process voice_prompt audio.
    # Only strip encoders if you exclusively use pre-encoded voices.

    processor = KugelAudioProcessor.from_pretrained(model_id)

    # Show available pre-encoded voices
    print(f"Available voices: {processor.get_available_voices()}")

    # ------------------------------------------------------------------ #
    # Example 1: Clone voice from an audio file (on-the-fly)               #
    # ------------------------------------------------------------------ #
    # Provide the path to a reference WAV file.  The processor sends the
    # raw audio to the model, which encodes it through its acoustic and
    # semantic tokenizers at inference time.
    #
    # Replace with a real path to your reference audio:
    reference_audio = "roland.wav"

    print("\n--- Example 1: Voice cloning from audio file ---")
    print(f"Reference audio: {reference_audio}")
    print("(Skipping -- replace 'reference_audio' with a real file to run)")

    # Uncomment the block below once you have a reference audio file:
    #
    inputs = processor(
         text="Hallo, jetzt kann ich mit Kugelaudio auch meine Stimme klonen. Ziemlich cool. ",
         voice_prompt=reference_audio,
         return_tensors="pt",
     )
    inputs = {
         k: v.to(device) if isinstance(v, torch.Tensor) else v
         for k, v in inputs.items()
     }
    
    with torch.no_grad():
         outputs = model.generate(**inputs, cfg_scale=3.0, max_new_tokens=512)
    #
    processor.save_audio(outputs.speech_outputs[0], "output_cloned.wav")
    print("Saved: output_cloned.wav")

    # ------------------------------------------------------------------ #
    # Example 2: Use named pre-encoded voices                              #
    # ------------------------------------------------------------------ #
    # The easiest way to change the speaker.  Available voices are listed
    # in voices/voices.json: "angry", "radio", "old_lady".

    print("\n--- Example 2a: Professional radio voice ---")
    inputs = processor(
        text="Guten Abend und herzlich willkommen zur heutigen Sendung. In den Nachrichten bringen wir Ihnen die neuesten Entwicklungen aus aller Welt.",
        voice="radio",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0, max_new_tokens=512, speech_end_penalty=5.0)

    processor.save_audio(outputs.speech_outputs[0], "output_radio.wav")
    print("Saved: output_radio.wav")

    print("\n--- Example 2b: Angry voice ---")
    inputs = processor(
        text="Das ist absolut inakzeptabel! Ich habe genug von diesem Unsinn. Es muss sich jetzt sofort etwas ändern!",
        voice="angry",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0, max_new_tokens=512, speech_end_penalty=5.0)

    processor.save_audio(outputs.speech_outputs[0], "output_angry.wav")
    print("Saved: output_angry.wav")

    print("\n--- Example 2c: Gentle elderly voice ---")
    inputs = processor(
        text="Ach mein Liebes, lass mich dir eine Geschichte aus meiner Jugend erzählen. Das waren so wundervolle Zeiten, voller Freude und Lachen.",
        voice="old_lady",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0, max_new_tokens=512, speech_end_penalty=5.0)

    processor.save_audio(outputs.speech_outputs[0], "output_old_lady.wav")
    print("Saved: output_old_lady.wav")

    # ------------------------------------------------------------------ #
    # Example 3: Encode a voice once, reuse many times                     #
    # ------------------------------------------------------------------ #
    # Use encode_voice_prompt() to pre-encode a reference audio file into
    # a voice_cache dict.  Save it as a .pt file and load it later to
    # avoid re-encoding on every generation.
    #
    # NOTE: The encoders must still be loaded (do not strip_encoders).

    print("\n--- Example 3: Encode & cache a custom voice ---")
    print("(Skipping -- replace 'reference_audio' with a real file to run)")

    # Uncomment the block below once you have a reference audio file:
    #
    voice_cache = model.encode_voice_prompt(reference_audio)
    #
    # Save for later reuse
    torch.save(voice_cache, "my_custom_voice.pt")
    print("Saved voice cache: my_custom_voice.pt")
    #
    # Load the cached voice (can be used without the encoders)
    voice_cache = torch.load("my_custom_voice.pt", map_location="cpu", weights_only=True)
    #
    inputs = processor(
         text="Und wenn ich will, kann ich diese Stimme auch immer wieder verwenden. Cool. ",
         voice_cache=voice_cache,
         return_tensors="pt",
     )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
     }
    #
    with torch.no_grad():
        outputs = model.generate(**inputs, cfg_scale=3.0, max_new_tokens=512, speech_end_penalty=5.0)
    #
    processor.save_audio(outputs.speech_outputs[0], "output_cached_voice.wav")
    print("Saved: output_cached_voice.wav")

    # ------------------------------------------------------------------ #
    # Tip: Creating voices from the command line                           #
    # ------------------------------------------------------------------ #
    # You can also create reusable voice files with the included utility:
    #
    #   python scripts/create_voice.py \
    #       --input speaker.wav \
    #       --output voices/my_voice.pt \
    #       --name "My Voice" \
    #       --description "Warm male narrator" \
    #       --language en

    print("\nDone.")


if __name__ == "__main__":
    main()
