"""Gradio web interface for KlonAudio text-to-speech."""

import os
import tempfile
import threading
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    warnings.warn("Gradio not installed. Install with: pip install gradio")


# Global model instances (lazy loaded)
_model = None
_processor = None
_watermark = None
_current_model_id = None  # Track which model is loaded


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_available_voices():
    """Get list of available pre-encoded voices from the loaded processor or voices.json."""
    global _processor
    if _processor is not None:
        return _processor.get_available_voices()

    # Fallback: load directly from voices.json if processor not loaded yet
    import json
    from pathlib import Path

    # Try to find voices.json in the package
    try:
        # First try relative to this file
        voices_path = Path(__file__).parent.parent / "voices" / "voices.json"
        if not voices_path.exists():
            # Try in current directory
            voices_path = Path("voices/voices.json")

        if voices_path.exists():
            with open(voices_path, "r") as f:
                voices_data = json.load(f)
                return list(voices_data.keys())
    except Exception as e:
        print(f"Warning: Could not load voices.json: {e}")

    return []


def _check_model_status():
    """Check if model is loaded and return status string."""
    global _model
    if _model is not None:
        return "**Status:** ✅ Model loaded and ready"
    else:
        return "**Status:** ⏳ Loading model in background..."


def _warmup_model(model, processor=None):
    """Warmup model components to eliminate CUDA kernel compilation overhead on first generation.

    This runs dummy data through all model components (acoustic decoder,
    diffusion head, language model) to trigger JIT compilation before actual inference.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        # 1. Warmup acoustic decoder (biggest impact - saves ~190ms on first call)
        latent_dim = model.config.acoustic_vae_dim
        dummy_latent = torch.randn(1, latent_dim, 1, device=device, dtype=dtype)
        _ = model.acoustic_tokenizer.decode(dummy_latent)

        # 2. Warmup diffusion/prediction head
        hidden_size = model.config.decoder_config.hidden_size
        model.noise_scheduler.set_timesteps(model.ddpm_inference_steps)

        dummy_condition = torch.randn(2, hidden_size, device=device, dtype=dtype)
        dummy_speech = torch.randn(2, latent_dim, device=device, dtype=dtype)

        for t in model.noise_scheduler.timesteps:
            half = dummy_speech[:1]
            combined = torch.cat([half, half], dim=0)
            _ = model.prediction_head(
                combined,
                t.repeat(combined.shape[0]).to(combined),
                condition=dummy_condition,
            )
            dummy_eps = torch.randn_like(dummy_speech)
            dummy_speech = model.noise_scheduler.step(dummy_eps, t, dummy_speech).prev_sample

        # 3. Warmup language model with KV cache path
        dummy_ids = torch.randint(0, 32000, (1, 64), device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        _ = model.model.language_model(
            input_ids=dummy_ids, attention_mask=dummy_mask, use_cache=True
        )

        # 4. Run a minimal generation to warmup the full generation path
        if processor is not None:
            dummy_inputs = processor(text="Hi.", return_tensors="pt")
            dummy_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in dummy_inputs.items()
            }
            _ = model.generate(
                **dummy_inputs, cfg_scale=3.0, max_new_tokens=10, show_progress=False
            )

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_models(model_id: str = "Roland-JAAI/klonaudio"):
    """Load model and processor. Switches model if a different model_id is requested."""
    global _model, _processor, _watermark, _current_model_id

    from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
    from kugelaudio_open.processors import KugelAudioProcessor
    from kugelaudio_open.watermark import AudioWatermark

    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Check if we need to load a different model
    if _model is None or _current_model_id != model_id:
        # Clean up old model if switching
        if _model is not None and _current_model_id != model_id:
            print(f"Switching model from {_current_model_id} to {model_id}...")
            del _model
            del _processor
            _model = None
            _processor = None
            # Clear CUDA cache to free memory
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"Loading model {model_id} on {device}...")
        try:
            _model = KugelAudioForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
            ).to(device)
        except Exception:
            _model = KugelAudioForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)
        _model.eval()
        # Note: Don't strip encoders yet - we may need them for voice cloning
        # They will be stripped on first generation with pre-encoded voices
        _current_model_id = model_id
        print(f"Model {model_id} loaded!")

    if _processor is None:
        _processor = KugelAudioProcessor.from_pretrained(model_id)

    # Warmup to eliminate first-generation slowness from CUDA kernel compilation
    # Do this after processor is loaded so we can run a mini-generation
    if device == "cuda" and _model is not None:
        # Check if we need to warmup (only on first load)
        if not getattr(_model, "_warmed_up", False):
            print("Warming up model (this may take a moment)...")
            _warmup_model(_model, _processor)
            _model._warmed_up = True
            print("Warmup complete!")

    if _watermark is None:
        _watermark = AudioWatermark(device=device)

    return _model, _processor, _watermark


def generate_speech(
    text: str,
    voice_name: Optional[str] = None,
    voice_audio: Optional[Tuple[int, np.ndarray]] = None,
    model_choice: str = "kugelaudio-0-open",
    cfg_scale: float = 3.0,
    max_tokens: int = 2048,
) -> Tuple[int, np.ndarray]:
    """Generate speech from text using a pre-encoded voice or voice cloning.

    Args:
        text: Text to synthesize
        voice_name: Name of a pre-encoded voice (from voices.json registry)
        voice_audio: Audio for voice cloning (sample_rate, audio_array)
        model_choice: Model variant to use
        cfg_scale: Classifier-free guidance scale
        max_tokens: Maximum generation tokens

    Returns:
        Tuple of (sample_rate, audio_array)

    Note:
        All generated audio is automatically watermarked for identification.
    """
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    model_id = f"kugelaudio/{model_choice}"
    # Reuse cached model if already loaded
    model, processor, watermark = load_models(model_id)
    device = next(model.parameters()).device
    print(f"[Generation] Using cached model: {_current_model_id}")

    # Process text input with voice (priority: voice_audio > voice_name > default)
    if voice_audio is not None:
        # Voice cloning from uploaded/recorded audio
        sr, audio_data = voice_audio

        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # Resample to 24kHz if needed (model expects 24kHz)
        if sr != 24000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
            print(f"[Voice Cloning] Resampled audio from {sr}Hz to 24000Hz")

        audio_tensor = torch.from_numpy(audio_data).float()
        inputs = processor(text=text.strip(), voice_prompt=audio_tensor, return_tensors="pt")
    elif voice_name and voice_name != "None":
        # Pre-encoded voice
        inputs = processor(text=text.strip(), voice=voice_name, return_tensors="pt")
    else:
        # Default voice
        inputs = processor(text=text.strip(), return_tensors="pt")

    # Move tensors to device, keep dicts as-is
    model_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)
        else:
            model_inputs[k] = v

    voice_source = "cloned" if voice_audio is not None else (voice_name if voice_name != "None" else "default")
    print(
        f"[Generation] Using model: {model_id}, voice={voice_source}, cfg_scale={cfg_scale}, max_tokens={max_tokens}"
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=max_tokens,
        )

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise gr.Error("Generation failed. Please try again with different settings.")

    # Audio is already watermarked by the model's generate method
    audio = outputs.speech_outputs[0]
    print(f"[Generation] Raw output: shape={audio.shape}, dtype={audio.dtype}")

    # Convert to numpy (convert to float32 first since numpy doesn't support bfloat16)
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()

    # Ensure correct shape (1D array)
    audio = audio.squeeze()

    # Normalize to prevent clipping (important for Gradio playback)
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val * 0.95

    print(
        f"[Generation] Final output: shape={audio.shape}, dtype={audio.dtype}, duration={len(audio)/24000:.2f}s"
    )
    print(
        f"[Generation] Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, std={audio.std():.4f}"
    )

    # Return with explicit sample rate - Gradio expects (sample_rate, audio_array)
    return (24000, audio)


def check_watermark(audio: Tuple[int, np.ndarray]) -> str:
    """Check if audio contains KugelAudio watermark."""
    if audio is None:
        return "No audio provided."

    from kugelaudio_open.watermark import AudioWatermark

    sr, audio_data = audio

    # Convert to float32 if needed
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0

    watermark = AudioWatermark()
    result = watermark.detect(audio_data, sample_rate=sr)

    if result.detected:
        return f"✅ **Watermark Detected**\n\nConfidence: {result.confidence:.1%}\n\nThis audio was generated by KlonAudio (based on KugelAudio)."
    else:
        return f"❌ **No Watermark Detected**\n\nConfidence: {result.confidence:.1%}\n\nThis audio does not appear to be generated by KlonAudio."


def create_app(preload_model: bool = True) -> "gr.Blocks":
    """Create the Gradio application.

    Args:
        preload_model: If True, load the model at startup instead of on first generation
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not installed. Install with: pip install gradio")

    # Model will be loaded in background after server starts if preload_model=True
    # This allows the UI to be accessible immediately

    # Logo URLs
    kugelaudio_logo = "https://www.kugelaudio.com/logos/Logo%20Short.svg"
    kisz_logo = "https://docs.sc.hpi.de/attachments/aisc/aisc-logo.png"
    bmftr_logo = (
        "https://hpi.de/fileadmin/_processed_/a/3/csm_BMFTR_de_Web_RGB_gef_durch_cd1f5345bd.jpg"
    )

    with gr.Blocks(
        title="KlonAudio - Text to Speech with Voice Cloning",
    ) as app:
        gr.HTML(
            """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.5rem;">🎙️ KlonAudio</h1>
            <p style="color: #666; margin-bottom: 0.5rem;">Open-source text-to-speech with full voice cloning</p>
            <p style="color: #888; font-size: 0.9rem;">
                Fork of <a href="https://github.com/Kugelaudio/kugelaudio-open" target="_blank" style="color: #667eea;">KugelAudio</a>
                with restored voice cloning capabilities
            </p>
        </div>
        """
        )

        # Model status indicator (updates every 2 seconds until model is loaded)
        model_status = gr.Markdown(
            value=_check_model_status,
            every=2,
            elem_id="model-status"
        )

        with gr.Tabs():
            # Tab 1: Text to Speech
            with gr.TabItem("🗣️ Generate Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5,
                            max_lines=20,
                        )

                        voice_dropdown = gr.Dropdown(
                            choices=["None"] + _get_available_voices(),
                            value="radio",
                            label="Pre-encoded Voice (Optional)",
                            info="Select a pre-encoded voice or use voice cloning below",
                        )

                        with gr.Accordion("🎤 Voice Cloning (Upload or Record)", open=False):
                            gr.Markdown(
                                """
                            Clone any voice by uploading a 5-10 second audio sample or recording your own voice.
                            **Note:** Voice cloning overrides the pre-encoded voice selection above.
                            """
                            )
                            voice_audio_input = gr.Audio(
                                label="Reference Audio for Voice Cloning",
                                type="numpy",
                                sources=["upload", "microphone"],
                            )
                            gr.Markdown(
                                """
                            **Tips for best results:**
                            - Use 5-10 seconds of clear speech
                            - Minimize background noise
                            - Speak naturally at normal volume
                            """
                            )

                        with gr.Accordion("Advanced Settings", open=False):
                            model_choice = gr.Dropdown(
                                choices=["kugelaudio-0-open"],
                                value="kugelaudio-0-open",
                                label="Model",
                            )
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.5,
                                label="Guidance Scale",
                                info="Higher values = more adherence to text",
                            )
                            max_tokens = gr.Slider(
                                minimum=512,
                                maximum=8192,
                                value=2048,
                                step=256,
                                label="Max Tokens",
                                info="Maximum generation length",
                            )

                        generate_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="Generated Speech",
                            type="numpy",
                            interactive=False,
                        )

                        gr.Markdown(
                            """
                        ### Tips
                        - For best results, use clear and well-punctuated text
                        - **Voice Cloning**: Upload or record audio for custom voices
                        - **Pre-encoded**: Select from dropdown for quick generation
                        - Leave both empty for default/random voice
                        """
                        )

                generate_btn.click(
                    fn=generate_speech,
                    inputs=[text_input, voice_dropdown, voice_audio_input, model_choice, cfg_scale, max_tokens],
                    outputs=[output_audio],
                )

            # Tab 2: Watermark Detection
            with gr.TabItem("🔍 Verify Watermark"):
                gr.Markdown(
                    """
                ### Watermark Verification
                Check if an audio file was generated by KlonAudio. All audio generated
                by KlonAudio contains an imperceptible watermark for identification.
                """
                )

                with gr.Row():
                    with gr.Column():
                        verify_audio = gr.Audio(
                            label="Audio to Verify",
                            type="numpy",
                            sources=["upload"],
                        )
                        verify_btn = gr.Button("🔍 Check Watermark", variant="secondary")

                    with gr.Column():
                        verify_result = gr.Markdown(
                            label="Result",
                            value="Upload an audio file to check for watermark.",
                        )

                verify_btn.click(
                    fn=check_watermark,
                    inputs=[verify_audio],
                    outputs=[verify_result],
                )

            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                ## About KlonAudio

                KlonAudio is a fork of KugelAudio with restored voice cloning capabilities. It combines:

                - **AR + Diffusion Architecture**: Uses autoregressive language modeling
                  with diffusion-based speech synthesis for high-quality output
                - **Voice Cloning**: Clone any voice with 5-10 seconds of reference audio
                  - Upload audio files or record directly in the browser
                  - Supports 23 European languages
                  - Dual-encoder architecture (acoustic + semantic) for high quality
                - **Audio Watermarking**: All generated audio contains an imperceptible watermark
                  using [Facebook's AudioSeal](https://huggingface.co/facebook/audioseal) technology
                
                ### Models
                
                | Model | Parameters | Quality | Speed |
                |-------|------------|---------|-------|
                | kugelaudio-0-open | 7B | Best | Standard |
                
                ### Responsible Use
                
                This technology is intended for legitimate purposes such as:
                - Accessibility (text-to-speech for visually impaired)
                - Content creation (podcasts, videos, audiobooks)
                - Voice assistants and chatbots
                
                **Please do not use this technology for:**
                - Creating deepfakes or misleading content
                - Impersonating individuals without consent
                - Any illegal or harmful purposes
                
                All generated audio is watermarked to enable detection.
                """
                )

        gr.HTML(
            """
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #eee;">
            <p style="color: #888; margin-bottom: 0.5rem;">
                <strong>KlonAudio</strong> • Open Source TTS with Voice Cloning
            </p>
            <p style="color: #aaa; font-size: 0.85rem;">
                Open Source Release by <a href="https://github.com/RolandJAAI" style="color: #667eea;" target="_blank">Roland Becker</a> •
                <a href="https://github.com/RolandJAAI/klonaudio" style="color: #667eea;" target="_blank">GitHub</a>
            </p>
            <p style="color: #bbb; font-size: 0.8rem; margin-top: 0.5rem;">
                Based on <a href="https://github.com/Kugelaudio/kugelaudio-open" style="color: #667eea;" target="_blank">KugelAudio</a>
                by <a href="mailto:kajo@kugelaudio.com" style="color: #667eea;">Kajo Kratzenstein</a> and
                <a href="https://github.com/microsoft/VibeVoice" style="color: #667eea;" target="_blank">Microsoft VibeVoice</a>
            </p>
        </div>
        """
        )

    return app


def launch_app(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    preload_model: bool = True,
    **kwargs,
):
    """Launch the Gradio web interface.

    Args:
        share: Create a public share link
        server_name: Server hostname (use "0.0.0.0" for network access)
        server_port: Server port
        preload_model: If True, load model in background after server starts
        **kwargs: Additional arguments passed to gr.Blocks.launch()
    """
    app = create_app(preload_model=preload_model)

    # Start model loading in background thread if preload_model is True
    if preload_model:
        def load_model_background():
            print("🔄 Loading model in background...")
            try:
                load_models("Roland-JAAI/klonaudio")
                print("✅ Model loaded and ready!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")

        # Start background thread
        loading_thread = threading.Thread(target=load_model_background, daemon=True)
        loading_thread.start()
        print("🚀 Server starting... Model will load in background.")

    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        **kwargs,
    )


if __name__ == "__main__":
    launch_app()
