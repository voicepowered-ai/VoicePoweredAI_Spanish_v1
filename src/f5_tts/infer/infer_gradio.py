# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "Spanish-F5"
tts_model_choice = DEFAULT_TTS_MODEL

# load models

vocoder = load_vocoder()

def load_f5tts(ckpt_path=str(cached_path("hf://VoicePoweredAI/VoicePoweredAI_Spanish_v1/spanish_v1/model_esp.safetensors"))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None

#SPAIN_ema_model, pre_new_tts_path = None, ""

@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    threshold,
    batch_size,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
    cfg_strength=2.0
):
    if not ref_audio_orig:
        gr.Warning("Debe subir un audio de referencia.")
        return gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Debe escribir un texto a generar.")
        return gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model

    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
        threshold=threshold,
        batch_size=batch_size,
        cfg_strength=cfg_strength
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave), ref_text


with gr.Blocks() as app_tts:
    gr.Markdown("# TTS por Fragmentos")
    ref_audio_input = gr.Audio(label="Audio de referencia", type="filepath")
    gen_text_input = gr.Textbox(label="Texto a generar", lines=10)
    generate_btn = gr.Button("Sintetizar", variant="primary")
    with gr.Accordion("Configuraciones Avanzadas", open=False):
        ref_text_input = gr.Textbox(
            label="Texto de Referencia",
            info="Deja en blanco para transcribir automáticamente el audio de referencia. Si ingresas texto, sobrescribirá la transcripción automática.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Eliminar Silencios",
            info="El modelo tiende a producir silencios, especialmente en audios más largos. Podemos eliminar manualmente los silencios si es necesario. Ten en cuenta que esta es una característica experimental y puede producir resultados extraños. Esto también aumentará el tiempo de generación.",
            value=True,
        )
        speed_slider = gr.Slider(
            label="Velocidad",
            minimum=0.3,
            maximum=2.0,
            value=1.01,
            step=0.01,
            info="Ajusta la velocidad del audio.",
        )
        cfg_strength_slider = gr.Slider(
            label="Intensidad de CFG",
            minimum=1.0,
            maximum=5.0,
            value=2.0,
            step=0.1,
            info="Controla la fuerza del Classifier-Free Guidance. Valores más altos producen audio más fiel al texto pero pueden reducir la naturalidad.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Duración del Cross-Fade (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Establece la duración del cross-fade entre clips de audio.",
        )
        steps = gr.Slider(
            label="Pasos de Inferencia",
            minimum=16,
            maximum=64,
            value=50,
            step=1,
            info="Aumentando los pasos de inferencia, se aumenta la calidad del audio de salida, pero también el tiempo que se tarda en generar el audio.",
        )
        threshold = gr.Slider(
            label="Umbral del Corrector",
            minimum=0,
            maximum=1,
            value=0.94,
            step=0.01,
            info="El umbral del corrector define la similitud que debe tener el audio de salida con el texto a generar, regenerando el fragmento de audio hasta 3 veces para maximizar este valor (mientras sea menor que el umbral).",
        )
        batch_size = gr.Slider(
            label="Duración de Cada Fragmento",
            minimum=5,
            maximum=15,
            value=15,
            step=0.5,
            info="Este valor define la longitud máxima de cada fragmento de audio sintetizado. Estos fragmentos se unen con un cross-fade para conseguir el audio final. Valor experimental.",
        )

    audio_output = gr.Audio(label="Audio Sintetizado")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
        threshold,
        batch_size,
        cfg_strength_slider
    ):
        audio_out, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            threshold,
            batch_size,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
            cfg_strength=cfg_strength_slider
        )
        return audio_out, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            steps,
            speed_slider,
            threshold,
            batch_size,
            cfg_strength_slider
        ],
        outputs=[audio_output, ref_text_input],
    )

    def clear_ref_text(audio):
        if audio is None:
            return gr.update(value="")
        return gr.update(value="")

    ref_audio_input.change(
        fn=clear_ref_text,
        inputs=[ref_audio_input],
        outputs=[ref_text_input]
    )


with gr.Blocks() as app_multistyle:
    gr.Markdown("# Generación Multi-Voz por Segmentos")

    # --- 1. Define Voices Section ---
    gr.Markdown("### Paso 1: Definir Voces")
    gr.Markdown(
        "Agrega las voces que usarás en tu guión. Proporciona un nombre único, un audio de referencia y, opcionalmente, el texto de referencia para cada voz. Las voces pueden ser de distintas personas o de la misma persona, y se pueden usar diferentes registros o emociones en cada una."
    )

    MAX_VOICES = 10 # Limit the number of voices for UI simplicity
    voices_state = gr.State([]) # Store [{'name': str, 'audio': str, 'ref_text': str}]

    voice_rows = []
    voice_components = [] # Store tuples: (name_input, audio_input, ref_text_input, row)
    voice_data_inputs = [] # Store just the input components: [name, audio, text, name, audio, text, ...]


    # Function to update the central voices_state, filter the script, and return UI updates
    def update_voices_state(visible_count, current_script_list, *all_voice_data_values):
        current_voices = []
        active_voice_names = set()
        num_components_per_voice = 3 # name, audio, ref_text

        for i in range(visible_count):
            idx = i * num_components_per_voice
            name = all_voice_data_values[idx]
            audio = all_voice_data_values[idx + 1]
            ref_text = all_voice_data_values[idx + 2]

            if name and audio:
                # Prevent using the reserved name "Pausa"
                if name == "Pausa":
                    gr.Warning("'Pausa' es un nombre de voz reservado. Por favor elige otro nombre.")
                    continue # Skip this voice definition

                if name in active_voice_names:
                    gr.Warning(f"Nombre de voz duplicado: '{name}'. Por favor, usa nombres únicos.")
                else:
                    current_voices.append({"name": name, "audio": audio, "ref_text": ref_text or ""})
                    active_voice_names.add(name)

        voice_names = [v["name"] for v in current_voices]
        default_voice = voice_names[0] if voice_names else None
        if not isinstance(current_script_list, list): current_script_list = []
        filtered_script_list = [seg for seg in current_script_list if seg.get("voice") == "Pausa" or seg.get("voice") in active_voice_names]
        return current_voices, filtered_script_list, gr.update(choices=voice_names or [], value=default_voice), json.dumps(filtered_script_list, indent=2, ensure_ascii=False)

    # Create rows for defining voices
    with gr.Column():
        for i in range(MAX_VOICES):
            is_initially_visible = (i == 0)
            with gr.Row(visible=is_initially_visible) as row:
                with gr.Column(scale=2):
                    name_input = gr.Textbox(
                        label="Nombre Voz" + (" (Obligatorio)" if i == 0 else ""),
                        value="Voz1" if i == 0 else "",
                        interactive=True
                    )
                with gr.Column(scale=3):
                     audio_input = gr.Audio(label="Audio Referencia", type="filepath", interactive=True)
                with gr.Column(scale=3):
                     ref_text_input = gr.Textbox(label="Texto Referencia (Opcional)", lines=1, interactive=True)
                with gr.Column(scale=1, min_width=80):
                    delete_btn = gr.Button("Eliminar", variant="secondary", visible=(i > 0))

            voice_rows.append(row)
            voice_components.extend([name_input, audio_input, ref_text_input, row])
            voice_data_inputs.extend([name_input, audio_input, ref_text_input])

    add_voice_btn = gr.Button("Agregar Otra Voz")

    # --- 2. Build Script Section (Manual Input) ---
    gr.Markdown("### Paso 2: Construir Guión")
    gr.Markdown(
        "Selecciona una voz, escribe el texto y haz clic en 'Agregar Segmento' para añadirlo al guión. Usa 'Añadir Pausa' para insertar silencios. El guión puede ser editado manualmente en el cuadro de texto."
    )

    # State to hold the script as a list of dictionaries
    script_list_state = gr.State([])
    # Add state for audio segment cache
    audio_cache_state = gr.State({}) # Stores {(voice, text): (sr, audio_data)}

    # Need initial values before defining components that use them
    initial_voice_names = ["Voz1"] if voice_rows[0].visible else []
    initial_default_voice = "Voz1" if initial_voice_names else None

    with gr.Row():
        segment_voice_select = gr.Dropdown(
            label="Voz para el Segmento",
            choices=initial_voice_names,
            value=initial_default_voice,
            interactive=True,
            allow_custom_value=False
        )
        segment_text_input = gr.Textbox(
            label="Texto del Segmento",
            lines=2,
            interactive=True
        )
        add_segment_button = gr.Button("Agregar Segmento", variant="secondary")

    # -- Pause Addition --
    with gr.Row():
        pause_duration_input = gr.Number(
            label="Duración Pausa (s)",
            value=0.5,
            minimum=0.1,
            step=0.1,
            interactive=True
        )
        add_pause_button = gr.Button("Añadir Pausa", variant="secondary")

    gr.Markdown("#### Guión Actual")
    # Use Textbox for display/editing
    script_display = gr.Textbox(
        label="Script (Editar con precaución - debe ser JSON válido)",
        lines=10,
        interactive=True,
        scale=2
    )

    # Add buttons for script management
    with gr.Row():
        delete_last_segment_button = gr.Button("Borrar Último Segmento")
        clear_script_button = gr.Button("Borrar Todo el Guión")

    # Function to delete the last segment
    def delete_last_segment_fn(current_script_list):
        if current_script_list and isinstance(current_script_list, list):
            current_script_list.pop()
        # Return updated state and formatted JSON string for display, ensuring ASCII is off
        return current_script_list, json.dumps(current_script_list, indent=2, ensure_ascii=False)

    # Function to clear the entire script
    def clear_script_fn():
        # Return empty list for state and empty JSON string for display
        return [], "[]"

    # Configure script management buttons
    delete_last_segment_button.click(
        delete_last_segment_fn,
        inputs=[script_list_state],
        outputs=[script_list_state, script_display]
    )
    clear_script_button.click(
        clear_script_fn,
        inputs=None,
        outputs=[script_list_state, script_display]
    )

    # Function to add a regular segment
    def add_segment_fn(current_script_list, voice, text):
        if not voice or not text or not text.strip():
            gr.Warning("Por favor, selecciona una voz y escribe texto para el segmento.")
            # Return unchanged state and display
            return current_script_list, json.dumps(current_script_list, indent=2, ensure_ascii=False)

        new_segment = {"voice": voice, "text": text.strip()}
        if not isinstance(current_script_list, list): current_script_list = []
        current_script_list.append(new_segment)
        # Use formatter for display
        return current_script_list, json.dumps(current_script_list, indent=2, ensure_ascii=False), gr.update(value="")

    # Configure the "Agregar Segmento" button
    add_segment_button.click(
        add_segment_fn,
        inputs=[script_list_state, segment_voice_select, segment_text_input],
        outputs=[script_list_state, script_display, segment_text_input]
    )

    # Function to add a pause segment
    def add_pause_fn(current_script_list, duration):
        if duration is None or not isinstance(duration, (int, float)) or duration <= 0:
            gr.Warning("Por favor, ingresa una duración de pausa válida (número > 0).")
            return current_script_list, json.dumps(current_script_list, indent=2, ensure_ascii=False)

        pause_segment = {"voice": "Pausa", "text": str(float(duration))}
        if not isinstance(current_script_list, list): current_script_list = []
        current_script_list.append(pause_segment)
        # Return updated state and display
        return current_script_list, json.dumps(current_script_list, indent=2, ensure_ascii=False)

    # Configure the "Añadir Pausa" button
    add_pause_button.click(
        add_pause_fn,
        inputs=[script_list_state, pause_duration_input],
        outputs=[script_list_state, script_display]
    )

    # Function to update the script state when the textbox is edited manually
    def update_script_from_textbox(script_text):
        try:
            parsed_list = json.loads(script_text)
            if isinstance(parsed_list, list):
                # Optional: Add deeper validation (e.g., check if items are dicts with keys)
                return parsed_list # Update the state if JSON is valid list
            else:
                gr.Warning("El texto editado no es una lista JSON válida.")
                return gr.update() # Do not update state
        except json.JSONDecodeError as e:
            gr.Warning(f"Error de sintaxis JSON al editar el script: {e}")
            return gr.update() # Do not update state

    # Add change handler for the script display textbox
    script_display.change(
        update_script_from_textbox,
        inputs=[script_display],
        outputs=[script_list_state] # Only update the state
    )

    # --- Dynamic Row Logic (Voice Definition) ---
    current_visible_voices = gr.State(1)

    def add_voice_row_fn(count):
        row_updates = [gr.update() for _ in voice_rows]
        if count < MAX_VOICES:
            row_updates[count] = gr.update(visible=True)
            count += 1
        else:
            gr.Warning(f"Máximo de {MAX_VOICES} voces alcanzado.")
            count = MAX_VOICES
        return [count] + row_updates

    add_voice_btn.click(
        add_voice_row_fn,
        inputs=[current_visible_voices],
        outputs=[current_visible_voices] + voice_rows
    )

    def delete_voice_row_fn():
        return gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)

    # Assign delete logic AND the subsequent state/UI updates
    for i in range(1, MAX_VOICES):
        delete_btn = voice_rows[i].children[-1].children[0]
        name_comp = voice_components[i*4]
        audio_comp = voice_components[i*4+1]
        text_comp = voice_components[i*4+2]
        delete_btn.click(
            delete_voice_row_fn,
            inputs=None,
            outputs=[voice_rows[i], name_comp, audio_comp, text_comp],
            api_name=False
        ).then(
            update_voices_state,
            inputs=[current_visible_voices, script_list_state] + voice_data_inputs,
            # Output script state, dropdown, and TEXTBOX display
            outputs=[voices_state, script_list_state, segment_voice_select, script_display],
            api_name=False
        )

    # --- Connect Changes to Update Dropdown and Filter Script ---
    for input_comp in voice_data_inputs:
         input_comp.change(
             update_voices_state,
             inputs=[current_visible_voices, script_list_state] + voice_data_inputs,
             # Output script state, dropdown, and TEXTBOX display
             outputs=[voices_state, script_list_state, segment_voice_select, script_display],
             api_name=False
         )

    # --- 3. Generation Section ---
    gr.Markdown("### Paso 3: Generar Audio")
    generate_multistyle_btn = gr.Button("Generar Audio del Guión", variant="primary")
    audio_output_multistyle = gr.Audio(label="Audio Sintetizado Final")

    @gpu_decorator
    def generate_scripted_speech(
        voices_data, script_list, current_audio_cache,
        progress=gr.Progress(track_tqdm=True)
    ):
        if not voices_data: return None, current_audio_cache
        if not script_list or not isinstance(script_list, list) or len(script_list) == 0: return None, current_audio_cache

        voice_lookup = {voice["name"]: voice for voice in voices_data}
        generated_audio_segments = []
        final_sample_rate = None
        used_cache_keys = set()
        if not isinstance(current_audio_cache, dict): current_audio_cache = {}
        DEFAULT_SR_FOR_PAUSE = 24000

        # Helper function for fade-out
        def apply_fade_out(audio_data, fade_samples):
            if fade_samples <= 0 or fade_samples > len(audio_data):
                return audio_data
            # Linear fade-out ramp from 1.0 to 0.0
            fade_out_ramp = np.linspace(1.0, 0.0, fade_samples, dtype=audio_data.dtype)
            audio_data[-fade_samples:] *= fade_out_ramp
            return audio_data

        num_segments = len(script_list)
        for i, segment in enumerate(tqdm.tqdm(script_list, desc="Generando segmentos")):
            voice_name = segment.get("voice")
            text = segment.get("text")
            is_pause = (voice_name == "Pausa")

            current_sr = None
            current_audio_data = None

            if is_pause:
                # --- Handle Pause Segment --- #
                try:
                    duration = float(text)
                    if duration <= 0: raise ValueError("Pause duration must be positive")
                    sr_for_pause = final_sample_rate if final_sample_rate is not None else DEFAULT_SR_FOR_PAUSE
                    if final_sample_rate is None: tqdm.tqdm.write(f"Segmento {i+1}: Pausa - Usando SR por defecto {sr_for_pause}Hz")
                    else: tqdm.tqdm.write(f"Segmento {i+1}: Pausa - {duration}s @ {sr_for_pause}Hz")
                    num_samples = int(duration * sr_for_pause)
                    current_audio_data = np.zeros(num_samples, dtype=np.float32)
                    # No need to set current_sr here, as it's silence
                except (ValueError, TypeError) as e:
                    tqdm.tqdm.write(f"Segmento {i+1}: Error procesando pausa ('{text}'): {e}. Saltando.")
                    continue
            else:
                # --- Handle Speech Segment --- #
                if not voice_name or not text:
                    tqdm.tqdm.write(f"Segmento {i+1} inválido. Saltando.")
                    continue

                cache_key = (voice_name, text)
                used_cache_keys.add(cache_key)

                if voice_name not in voice_lookup:
                    tqdm.tqdm.write(f"Voz '{voice_name}' no encontrada. Saltando segmento {i+1}.")
                    continue
                voice_info = voice_lookup[voice_name]
                ref_audio_path = voice_info["audio"]
                ref_text = voice_info["ref_text"]
                if not ref_audio_path or not isinstance(ref_audio_path, str):
                    tqdm.tqdm.write(f"Audio de referencia inválido para '{voice_name}'. Saltando segmento {i+1}.")
                    continue

                # Check cache
                if cache_key in current_audio_cache:
                    try:
                        sr_cache, audio_data_cache = current_audio_cache[cache_key]
                        if sr_cache is not None and isinstance(audio_data_cache, np.ndarray):
                            current_sr, current_audio_data = sr_cache, audio_data_cache
                            tqdm.tqdm.write(f"Segmento {i+1} ('{voice_name}'): Cache HIT")
                        else:
                           tqdm.tqdm.write(f"... Cache HIT - Invalid data, regenerating...")
                           del current_audio_cache[cache_key]
                    except Exception as e:
                         tqdm.tqdm.write(f"... Error reading from cache ({e}), regenerating...")
                         if cache_key in current_audio_cache: del current_audio_cache[cache_key]

                # Generate if cache miss
                if current_audio_data is None:
                    tqdm.tqdm.write(f"Segmento {i+1} ('{voice_name}'): Cache MISS - Generando...")
                    try:
                        audio_out, _ = infer(
                            ref_audio_path, ref_text, text, DEFAULT_TTS_MODEL,
                            remove_silence=True,
                            threshold=0.94, batch_size=15, cross_fade_duration=0.15, nfe_step=50,
                            speed=1.01, cfg_strength=2.0, show_info=print,
                        )
                        if audio_out is not None and isinstance(audio_out, tuple) and len(audio_out) == 2:
                            sr_gen, audio_data_gen = audio_out
                            if sr_gen is not None and isinstance(audio_data_gen, np.ndarray):
                                current_sr, current_audio_data = sr_gen, audio_data_gen
                                current_audio_cache[cache_key] = (current_sr, current_audio_data)
                                tqdm.tqdm.write(f"... Generated and cached.")
                            else:
                                tqdm.tqdm.write(f"... Infer invalid output. Skipping.")
                                continue
                        else:
                            tqdm.tqdm.write(f"... Infer failed. Skipping.")
                            continue
                    except Exception as e:
                        error_message = f"Error generando segmento {i+1} para voz '{voice_name}': {e}"
                        print(error_message)
                        gr.Error(f"Error al generar el segmento {i+1}...")
                        return None, current_audio_cache

            # --- Process the generated/retrieved/silence segment --- #
            if current_audio_data is not None and isinstance(current_audio_data, np.ndarray):
                # Set sample rate from the first non-pause segment
                if not is_pause and final_sample_rate is None:
                    final_sample_rate = current_sr
                    tqdm.tqdm.write(f"Sample rate establecido a {final_sample_rate}Hz por segmento {i+1}.")
                elif not is_pause and final_sample_rate != current_sr:
                     gr.Warning(f"Inconsistencia en Sample Rate (esperado {final_sample_rate}, obtenido {current_sr})... Intentando continuar.")
                     # Consider resampling here if necessary

                # --- Apply Fade-Out IF current is speech AND next is pause --- #
                if not is_pause and (i + 1 < num_segments) and script_list[i+1].get("voice") == "Pausa":
                    if final_sample_rate: # Need sample rate for fade duration
                        fade_duration_s = 0.1
                        fade_samples = int(fade_duration_s * final_sample_rate)
                        tqdm.tqdm.write(f"Segmento {i+1}: Aplicando fade-out de {fade_duration_s}s antes de la pausa.")
                        current_audio_data = apply_fade_out(current_audio_data, fade_samples)
                    else:
                         tqdm.tqdm.write(f"Segmento {i+1}: No se pudo aplicar fade-out (SR no establecido aún)." )

                # Append the processed audio data
                generated_audio_segments.append(current_audio_data)
            else:
                tqdm.tqdm.write(f"Segmento {i+1} no produjo audio válido. Saltando apilación.")
                continue

        # --- Cache Cleanup --- #
        current_keys = set(current_audio_cache.keys())
        keys_to_delete = current_keys - used_cache_keys
        if keys_to_delete:
            print(f"Cache Cleanup: Removing {len(keys_to_delete)} unused segment(s)...")
            for key in keys_to_delete:
                del current_audio_cache[key]
                # Optional detailed logging:
                # print(f"  - Removed: {key[0]} - '{key[1][:20]}...'")
        else:
             print("Cache Cleanup: No unused segments found.")

        # --- Concatenate Audio --- #
        if generated_audio_segments:
            # Make sure we have a sample rate if only pauses were generated (edge case)
            if final_sample_rate is None: final_sample_rate = DEFAULT_SR_FOR_PAUSE

            valid_segments = [seg for seg in generated_audio_segments if isinstance(seg, np.ndarray) and seg.size > 0]
            if not valid_segments: return None, current_audio_cache
            try:
                final_audio_data = np.concatenate(valid_segments)
                return (final_sample_rate, final_audio_data), current_audio_cache
            except Exception as e:
                 print(f"Error concatenando segmentos: {e}")
                 gr.Error("Error al unir los segmentos...")
                 return None, current_audio_cache
        else:
            gr.Warning("No se generó ningún segmento...")
            # Return None for audio AND the cleaned cache
            return None, current_audio_cache

    # Update the generate button click inputs and outputs
    generate_multistyle_btn.click(
        generate_scripted_speech,
        inputs=[
            voices_state,
            script_list_state,
            audio_cache_state, # Add cache as input
        ],
        # Add cache as output
        outputs=[audio_output_multistyle, audio_cache_state],
        api_name="generate_multivoice_script"
    )


with gr.Blocks() as app:
    gr.Markdown(
        """
# VoicePowered v1

Esta es una interfaz web para VoicePowered v1, un modelo Open Source de TTS centrado en la generación de audio en castellano con acento de España.

Para los mejores resultados, asegúrate que la duración del audio de referencia esté entre 11 y 14 segundos, que comience y acabe con un pequeño silencio, y a ser posible que abarque frases completas.
El audio generado tendrá el mismo tono, registro y velocidad de habla que el audio de referencia, por lo que es importante que el audio de referencia sea de buena calidad y represente las características deseadas del audio generado.
"""
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle],
        ["TTS Básico", "Multi-Voz"]
    )

    # --- Footer --- #
    with gr.Row(elem_id="footer-row", variant="panel"):
        # Left Spacer Column
        with gr.Column(scale=1, min_width=100):
            pass

        # Center Content Column (containing both image and text)
        with gr.Column(scale=2, elem_id="footer-center-col"):
            gr.HTML(
                """
                <a href="https://www.voicepowered.ai/" target="_blank" style="display: inline-block; line-height: 0;">
                    <img src="/file=src/media/logo.png" alt="Voice Powered AI Logo"
                        style="width: 250px; height: auto; border: none; display: block; margin: 0 auto;">
                </a>
                """
            )
            gr.Markdown("© 2025 Voice Powered AI. Created with ❤", elem_classes=["footer-text"])

        # Right Spacer Column
        with gr.Column(scale=1, min_width=100):
            pass

    # Add CSS
    app.css = """
    #footer-row {
        /* Centers the 3 columns horizontally */
        justify-content: center;
        /* Vertically aligns the content of the columns */
        align-items: center;
        padding: 15px 10px; /* Added a bit more vertical padding */
        min-height: 80px; /* Ensure footer has some height */
    }

    #footer-center-col {
        /* Horizontally center the content *within* this column */
        text-align: center;
    }

    /* Target the paragraph generated by gr.Markdown */
    #footer-center-col .footer-text p {
        margin: 5px 0 0 0; /* Small margin top, no margin bottom */
        line-height: normal; /* Keep default line height */
        font-size: 0.9em; /* Optional: slightly smaller text */
        color: #555; /* Optional: slightly muted color */
    }

    /* Style the link specifically if needed */
    #footer-center-col a {
        display: inline-block; /* Treat the link as a block for centering */
        margin-bottom: 5px; /* Add a small space between logo and text */
    }

    /* Ensure image itself doesn't have odd alignment */
    #footer-center-col img {
        vertical-align: middle; /* Good practice */
    }
    """
        

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
def main(port, host, share, api, root_path):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api, root_path=root_path, allowed_paths=["src/media/logo.png"])


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
		
