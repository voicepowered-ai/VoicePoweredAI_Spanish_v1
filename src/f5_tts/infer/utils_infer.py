# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys

os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"../../{os.path.dirname(os.path.abspath(__file__))}/third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

import tensorflow as tf
import regex as re
from thefuzz import fuzz

from num2words import num2words

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin
)

import subprocess

_ref_audio_cache = {}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 50  # Changed from 32 to 50
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.01  # Changed from 1.0 to 1.01

# -----------------------------------------


def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado) 

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

    return texto_traducido


def preprocess_text(text):
    text = text.lower()

    text = text.replace('"', "'")
    text = text.replace("`","'")
    text = text.replace("´","'")
    text = text.replace("-"," ")
    text = text.replace("_"," ")
    text = text.replace("¿"," ")
    text = text.replace("¡"," ")
    text = text.replace("%"," por ciento ")
    text = text.replace("&"," y ")
    text = text.replace("*"," por ")
    text = text.replace("("," ")
    text = text.replace(")"," ")
    text = text.replace("="," igual ")
    text = text.replace("+"," mas ")
    text = text.replace("-"," menos ")
    text = text.replace("|"," o ")
    text = text.replace("/"," entre ")
    text = text.replace("^"," elevado a ")
    text = text.replace("~"," aproximadamente ")

    text = re.sub(r"([.,])(?=[^\s])", r"\1 ", text)
    text = traducir_numero_a_texto(text)

    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    

    return text


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device):
    if vocoder_name == "vocos":
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    #load locally from ckpts/asr/model.safetensors
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def apply_audio_processing(
    input_path, 
    output_path=None, 
    compression=True,
    compression_threshold=-15,
    compression_ratio=2,
    compression_attack=2000,
    compression_release=3700,
    compression_makeup=1,
    loudnorm=True,
    rubberband=False,
    rubberband_transients=512,
    rubberband_detector=2048,
    rubberband_smoothing=8388608,
    rubberband_window=2097152,
    rubberband_pitchq=67108864,
    volume=-2
):
    """
    Apply audio compression using ffmpeg
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file (optional)
        compression: Whether to apply compression
        compression_threshold: Compression threshold in dB
        compression_ratio: Compression ratio
        compression_attack: Attack time in microseconds
        compression_release: Release time in microseconds
        compression_makeup: Makeup gain
        loudnorm: Whether to apply loudness normalization
        rubberband: Whether to apply rubberband processing
        rubberband_transients: Transients parameter for rubberband
        rubberband_detector: Detector size for rubberband
        rubberband_smoothing: Smoothing parameter for rubberband
        rubberband_window: Window size for rubberband
        rubberband_pitchq: Pitch quality for rubberband
        volume: Volume adjustment in dB
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_output = temp_file.name
            
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path
        ]
        
        # Build filter chain properly
        filters = []
        if compression:
            filters.append(f"acompressor=threshold={compression_threshold}dB:ratio={compression_ratio}:"
                         f"attack={compression_attack}:release={compression_release}:"
                         f"makeup={compression_makeup}")
        if rubberband:
            filters.append(f"rubberband=transients={rubberband_transients}:"
                         f"detector={rubberband_detector}:smoothing={rubberband_smoothing}:"
                         f"window={rubberband_window}:pitchq={rubberband_pitchq}")
        if loudnorm:
            filters.append("loudnorm")

        filters.append(f"volume={volume}dB")

        print(filters)
        
        if filters:
            ffmpeg_cmd.extend(["-af", ",".join(filters)])
            
        ffmpeg_cmd.extend(["-ar", "24000", temp_output])

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        with open(temp_output, 'rb') as f:
            compressed_data = f.read()
        with open(output_path or input_path, 'wb') as f:
            f.write(compressed_data)
        
        print("path: ", temp_output)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error applying compression: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error during compression: {str(e)}")
        return False


def preprocess_ref_audio_text(
    ref_audio_orig, 
    ref_text, 
    clip_short=True, 
    show_info=print, 
    device=device,
    compression=False,  # Changed from True to False
    compression_threshold=-15,
    compression_ratio=2,
    compression_attack=2000,
    compression_release=3700,
    compression_makeup=1,
    loudnorm=True,
    rubberband=False,
    rubberband_transients=512,
    rubberband_detector=2048,
    rubberband_smoothing=8388608,
    rubberband_window=2097152,
    rubberband_pitchq=67108864,
    volume=0  # Changed from -4 to 0
):
    show_info("Converting audio...")
    temp_files = []  # Keep track of temp files
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_files.append(f.name)
            aseg = AudioSegment.from_file(ref_audio_orig)

            if clip_short:
                # 1. try to find long silence for clipping
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 20000:
                        show_info("Audio is over 15s, clipping short. (1)")
                        break
                    non_silent_wave += non_silent_seg

                # 2. try to find short silence for clipping if 1. failed
                if len(non_silent_wave) > 20000:
                    non_silent_segs = silence.split_on_silence(
                        aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                    )
                    non_silent_wave = AudioSegment.silent(duration=0)
                    for non_silent_seg in non_silent_segs:
                        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 20000:
                            show_info("Audio is over 15s, clipping short. (2)")
                            break
                        non_silent_wave += non_silent_seg

                aseg = non_silent_wave

                # 3. if no proper silence found for clipping
                if len(aseg) > 20000:
                    aseg = aseg[:20000]
                    show_info("Audio is over 15s, clipping short. (3)")

            aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            aseg.export(f.name, format="wav")
            
            # Apply compression to reference audio and get the processed file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as processed_f:
                temp_files.append(processed_f.name)
                apply_audio_processing(
                    f.name, 
                    processed_f.name,
                    compression=compression,
                    compression_threshold=compression_threshold,
                    compression_ratio=compression_ratio,
                    compression_attack=compression_attack,
                    compression_release=compression_release,
                    compression_makeup=compression_makeup,
                    loudnorm=loudnorm,
                    rubberband=rubberband,
                    rubberband_transients=rubberband_transients,
                    rubberband_detector=rubberband_detector,
                    rubberband_smoothing=rubberband_smoothing,
                    rubberband_window=rubberband_window,
                    rubberband_pitchq=rubberband_pitchq,
                    volume=volume
                )
                ref_audio = processed_f.name

        # Compute a hash of the reference audio file
        with open(ref_audio, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_hash = hashlib.md5(audio_data).hexdigest()

        if not ref_text.strip():
            global _ref_audio_cache
            if audio_hash in _ref_audio_cache:
                # Use cached asr transcription
                show_info("Using cached reference text...")
                ref_text = _ref_audio_cache[audio_hash]
            else:
                show_info("No reference text provided, transcribing reference audio...")
                ref_text = transcribe(ref_audio)
                # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
                _ref_audio_cache[audio_hash] = ref_text
        else:
            show_info("Using custom reference text...")

        ref_text = preprocess_text(ref_text)

        # Ensure ref_text ends with a proper sentence-ending punctuation
        if not ref_text.endswith(". "):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        print("\nref_text  ", ref_text)

        return ref_audio, ref_text
    except Exception as e:
        raise e
    finally:
        # Clean up all temp files except ref_audio which is still needed
        for temp_file in temp_files[:-1]:  # Keep the last file (ref_audio)
            try:
                os.unlink(temp_file)
            except:
                pass


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    threshold,
    batch_size,
    mel_spec_type="vocos",
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    device=device,
    compression=True,
    compression_threshold=-15,
    compression_ratio=2,
    compression_attack=2000,
    compression_release=3700,
    compression_makeup=1,
    loudnorm=True,
    rubberband=False,
    rubberband_transients=512,
    rubberband_detector=2048,
    rubberband_smoothing=8388608,
    rubberband_window=2097152,
    rubberband_pitchq=67108864,
    volume=-4
):
    
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text, show_info=show_info, device=device)

    gen_text=preprocess_text(gen_text)
    gen_text = re.sub(r'\n+', '\n', gen_text)
    gen_text = gen_text.replace("\n", "... ")
    gen_text = re.sub(r'\.{4,}', '...', gen_text)
    
    audio, sr = torchaudio.load(ref_audio)
    audio_duration = audio.shape[-1] / sr
    print(f"\n\nReference audio duration in seconds: {audio_duration} \n\n")
    speech_rate = len(ref_text) / audio_duration
    max_chars1 = int(speech_rate * batch_size)
    max_chars2 = int(batch_size / 0.06911)
    max_chars = (max_chars1 * 0.6 + max_chars2 * 0.4) 
    print(f"batch_size: {batch_size}")
    print(f"max_chars1: {max_chars1}, max_chars2: {max_chars2}, max_chars: {max_chars}")

    # Split the input text into batches
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    
    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process(
        (audio, sr),
        audio_duration,
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        device=device,
        threshold=threshold,
        compression=compression,
        compression_threshold=compression_threshold,
        compression_ratio=compression_ratio,
        compression_attack=compression_attack,
        compression_release=compression_release,
        compression_makeup=compression_makeup,
        loudnorm=loudnorm,
        rubberband=rubberband,
        rubberband_transients=rubberband_transients,
        rubberband_detector=rubberband_detector,
        rubberband_smoothing=rubberband_smoothing,
        rubberband_window=rubberband_window,
        rubberband_pitchq=rubberband_pitchq,
        volume=-volume
    )


def normalize_and_compare(asr_result, gen_text):
    asr_norm = re.sub(r"[^a-zA-Z0-9\sáéíóúüñ]", "", preprocess_text(asr_result).strip())
    gen_norm = re.sub(r"[^a-zA-Z0-9\sáéíóúüñ]", "", preprocess_text(gen_text).strip())
    return fuzz.ratio(asr_norm, gen_norm)


def calculate_duration_samples(text_length, target_sample_rate, hop_length, speech_rate=1.0):
    """
    Calculate the duration in samples using an exponential scaling approach.
    
    Args:
        text_length (int): Length of the text in bytes/chars
        target_sample_rate (int): Audio sample rate
        hop_length (int): Hop length for mel spectrogram
        speech_rate (float): Multiplier to adjust speech rate (default: 1.0)
    
    Returns:
        int: Duration in samples
    """
    base_char_weight = 0.066  # Base weight for normal-length texts
    
    duration_samples = int(text_length * base_char_weight * target_sample_rate / hop_length)
    
    # Ensure minimum duration of 1 second
    min_duration_samples = int(1 * target_sample_rate / hop_length)
    
    return max(duration_samples, min_duration_samples)


def trim_silence(wave, sample_rate, silence_threshold=-40, chunk_size=10):
    """
    Trims excessive silence while preserving 0.5s of original silence on both ends
    """
    # Save original properties
    original_dtype = wave.dtype
    original_shape = wave.shape

    # Convert to proper format for pydub processing
    if np.issubdtype(original_dtype, np.floating):
        pcm_wave = (wave * 32767).astype(np.int16)
    else:
        pcm_wave = wave.astype(np.int16)

    # Create AudioSegment
    audio = AudioSegment(
        pcm_wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

    preserve_samples_ms = 500

    # Find trim points
    silence_threshold_dBFS = silence_threshold
    chunk_size_ms = chunk_size
    
    # Find start trim point
    start_trim = preserve_samples_ms  # Start after preserved silence
    while start_trim < len(audio) - preserve_samples_ms:
        chunk = audio[start_trim:start_trim+chunk_size_ms]
        if chunk.dBFS > silence_threshold_dBFS:
            break
        start_trim += chunk_size_ms

    # Find end trim point
    end_trim = len(audio) - preserve_samples_ms  # Start before preserved silence
    while end_trim > preserve_samples_ms:
        chunk = audio[max(0, end_trim-chunk_size_ms):end_trim]
        if chunk.dBFS > silence_threshold_dBFS:
            break
        end_trim -= chunk_size_ms

    # Apply trimming with preserved silence
    trimmed = audio[max(0, start_trim-preserve_samples_ms):min(len(audio), end_trim+preserve_samples_ms)]

    # Convert back to numpy
    samples = np.array(trimmed.get_array_of_samples(), dtype=np.int16)

    # Restore original format
    if np.issubdtype(original_dtype, np.floating):
        samples = samples.astype(np.float32) / 32768.0

    # Preserve multi-channel format if needed
    if len(original_shape) > 1:
        return samples.reshape(-1, original_shape[1]).astype(original_dtype)
    
    return samples.astype(original_dtype)

# infer batches


def infer_batch_process(
    ref_audio,
    ref_audio_dur,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    threshold,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=50,  
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1.01, 
    device=None,
    compression=True,
    compression_threshold=-40, 
    compression_ratio=3,  
    compression_attack=2,  
    compression_release=250,  
    compression_makeup=1,
    loudnorm=True,
    rubberband=True,  
    rubberband_transients=512,
    rubberband_detector=2048,
    rubberband_smoothing=8388608,
    rubberband_window=2097152,
    rubberband_pitchq=67108864,
    volume=3  
):
    temp_files = []  # Keep track of temp files
    try:
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)
        generated_waves = []
        spectrograms = []

        for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
            text_list = [(ref_text.strip() + " " + gen_text.strip())]
            final_text_list = convert_char_to_pinyin(text_list)
            ref_audio_len = audio.shape[-1] // hop_length

            if not gen_text.endswith(". "):
                if gen_text.endswith("."):
                    gen_text += " "
                else:
                    gen_text += ". "

            if not gen_text.startswith(" "):
                gen_text = " " + gen_text

            gen_text_len = len(gen_text.encode("utf-8"))
            duration_samples = calculate_duration_samples(
                gen_text_len, 
                target_sample_rate, 
                hop_length,
                speech_rate=speed
            )
            
            duration1 = ref_audio_len + duration_samples
            ref_text_len = len(ref_text.encode("utf-8"))
            duration2 = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

            duration = (int) (duration1 * 0.6 + duration2 * 0.4 * 1/speed)

            print(f"duration 1: {duration1}, duration 2: {duration2}, duration: {duration}")

            attempts = 0
            max_attempts = 3
            best_ratio = 0
            best_candidate_wave = None

            print(f"Generating text {gen_text}")

            while attempts < max_attempts:
                attempts += 1
                with torch.inference_mode():
                    generated, _ = model_obj.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )
                    generated = generated.to(torch.float32)
                    generated = generated[:, ref_audio_len:, :]
                    generated_mel_spec = generated.permute(0, 2, 1)

                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(generated_mel_spec)
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(generated_mel_spec)

                    if rms < target_rms:
                        generated_wave = generated_wave * rms / target_rms

                    generated_wave = generated_wave.squeeze().cpu().numpy()
                    # Add silence trimming with 0.5s padding
                    generated_wave = trim_silence(
                        generated_wave, 
                        target_sample_rate,
                        silence_threshold=-40
                    )
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_files.append(temp_file.name)
                    torchaudio.save(temp_file.name, torch.tensor(generated_wave).unsqueeze(0), target_sample_rate)

                    if asr_pipe is None:
                        initialize_asr_pipeline(device=device)

                    asr_result = asr_pipe(temp_file.name)["text"].lower().strip()
                    current_ratio = normalize_and_compare(asr_result, gen_text.strip())

                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_candidate_wave = generated_wave

                    if best_ratio >= threshold:
                        break

            final_chunk = best_candidate_wave
            generated_waves.append(final_chunk)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        if cross_fade_duration <= 0:
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]
                cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
                new_wave = np.concatenate(
                    [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )
                final_wave = new_wave

        combined_spectrogram = np.concatenate(spectrograms, axis=1)

        # Process the final generated audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
            temp_files.append(temp_f.name)
            # Save the initial wave
            torchaudio.save(temp_f.name, torch.tensor(final_wave).unsqueeze(0), target_sample_rate)
            
            # Create another temp file for processed audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as processed_f:
                temp_files.append(processed_f.name)
                # Apply audio processing
                apply_audio_processing(
                    temp_f.name, 
                    processed_f.name,
                    compression=compression,
                    compression_threshold=compression_threshold,
                    compression_ratio=compression_ratio,
                    compression_attack=compression_attack,
                    compression_release=compression_release,
                    compression_makeup=compression_makeup,
                    loudnorm=loudnorm,
                    rubberband=rubberband,
                    rubberband_transients=rubberband_transients,
                    rubberband_detector=rubberband_detector,
                    rubberband_smoothing=rubberband_smoothing,
                    rubberband_window=rubberband_window,
                    rubberband_pitchq=rubberband_pitchq,
                    volume=-volume
                )
                # Load the processed audio
                processed_wave, sr = torchaudio.load(processed_f.name)
                final_wave_processed = processed_wave.squeeze().numpy()

        return final_wave_processed, target_sample_rate, combined_spectrogram
    finally:
        # Clean up all temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

# remove silence from generated wav

def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()