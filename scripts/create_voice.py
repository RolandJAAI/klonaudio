#!/usr/bin/env python3
"""Create a reusable voice file from an audio recording.

This utility encodes an audio file into a voice embedding (.pt file) that can be
used for voice cloning with KugelAudio. The output file contains pre-encoded
acoustic and semantic features that bypass the need to re-encode the reference
audio on every generation.

Usage:
    python scripts/create_voice.py --input speaker.wav --output voices/my_voice.pt
    python scripts/create_voice.py --input speaker.wav --output voices/my_voice.pt \\
        --name "My Voice" --description "Warm male narrator" --language en

Then use the voice file for generation:
    from kugelaudio_open import KugelAudioForConditionalGenerationInference
    model = KugelAudioForConditionalGenerationInference.from_pretrained(...)
    voice_cache = model.load_voice("voices/my_voice.pt")
    outputs = model.generate(text_ids=..., voice_cache=voice_cache)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode an audio file into a reusable voice embedding (.pt file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic usage
  python scripts/create_voice.py --input speaker.wav --output my_voice.pt

  # With metadata
  python scripts/create_voice.py \\
      --input speaker.wav \\
      --output voices/narrator.pt \\
      --name "Narrator" \\
      --description "Deep male narrator voice" \\
      --language en

  # Custom model
  python scripts/create_voice.py \\
      --input speaker.wav \\
      --output my_voice.pt \\
      --model /path/to/local/model
""",
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input audio file (WAV, MP3, FLAC, etc.).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for the output .pt voice file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="kugelaudio/kugelaudio-0-open",
        help="Model ID or path to load (default: kugelaudio/kugelaudio-0-open).",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Human-readable name for this voice (default: derived from filename).",
    )
    parser.add_argument(
        "--description",
        "-d",
        default="",
        help="Description of the voice characteristics.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="en",
        help="Language code for the voice (default: en).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input file
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate output path
    output_path = Path(args.output).resolve()
    if not output_path.suffix:
        output_path = output_path.with_suffix(".pt")
    elif output_path.suffix != ".pt":
        print(
            f"Warning: Output file does not have .pt extension: {output_path.name}",
            file=sys.stderr,
        )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive voice name from filename if not provided
    voice_name = args.name if args.name else input_path.stem

    # --- Load model ---
    import torch
    from kugelaudio_open import KugelAudioForConditionalGenerationInference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Voice Creation Utility")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {args.model}")
    print(f"  Device: {device}")
    print()

    print("Loading model (this may take a moment)...")
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.set_ddpm_inference_steps()

    # --- Encode voice ---
    print(f"Encoding voice from: {input_path.name}")
    voice_cache = model.encode_voice_prompt(str(input_path))

    # --- Add metadata ---
    voice_cache["name"] = voice_name
    voice_cache["description"] = args.description
    voice_cache["language"] = args.language
    voice_cache["source_file"] = input_path.name
    voice_cache["created_at"] = datetime.now(timezone.utc).isoformat()
    voice_cache["model_id"] = args.model

    # --- Save ---
    print(f"Saving voice file to: {output_path}")
    torch.save(voice_cache, output_path)

    # --- Verify the saved file can be loaded ---
    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    assert "acoustic_mean" in loaded, "Verification failed: acoustic_mean not found in saved file"
    assert "semantic_mean" in loaded, "Verification failed: semantic_mean not found in saved file"

    # --- Print summary ---
    acoustic_shape = loaded["acoustic_mean"].shape
    semantic_shape = loaded["semantic_mean"].shape
    file_size_kb = output_path.stat().st_size / 1024

    print()
    print("Voice file created successfully!")
    print()
    print(f"  Name:            {voice_name}")
    print(f"  Description:     {args.description or '(none)'}")
    print(f"  Language:         {args.language}")
    print(f"  Source:           {input_path.name}")
    print(f"  Acoustic shape:  {list(acoustic_shape)}")
    print(f"  Semantic shape:  {list(semantic_shape)}")
    print(f"  File size:       {file_size_kb:.1f} KB")
    print()
    print("Usage:")
    print()
    print("  # Python API")
    print("  from kugelaudio_open import KugelAudioForConditionalGenerationInference")
    print(f'  voice_cache = KugelAudioForConditionalGenerationInference.load_voice("{output_path.name}")')
    print("  outputs = model.generate(text_ids=..., voice_cache=voice_cache)")
    print()
    print(f'  # Command line')
    print(f'  python start.py generate "Hello world" -r {output_path} -o output.wav')


if __name__ == "__main__":
    main()
