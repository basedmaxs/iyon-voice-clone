import gradio as gr
import torch
from f5_tts.f5_tts import F5TTS
import os
from pydub import AudioSegment

tts = F5TTS(device="cpu")  # Railway biasanya pakai CPU

def generate_speech(text, audio_file, speed=1.0):
    if not audio_file:
        return None, "❌ Upload suara referensi terlebih dahulu!"
    
    output_wav = "output.wav"
    output_mp3 = "output.mp3"
    
    try:
        tts.infer(
            text=text,
            ref_audio=audio_file,
            output=output_wav,
            speed=speed
        )
        
        # Convert ke MP3
        audio = AudioSegment.from_wav(output_wav)
        audio.export(output_mp3, format="mp3")
        
        return output_wav, output_mp3, "✅ Berhasil dibuat!"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

with gr.Blocks(title="Voice Cloning TTS Indonesia") as demo:
    gr.Markdown("# 🎙️ F5-TTS Voice Cloning")
    gr.Markdown("Upload suara referensi → Tulis teks → Generate")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Suara Referensi (5-20 detik)")
            text_input = gr.Textbox(lines=5, label="Teks yang ingin diucapkan", placeholder="Masukkan teks bahasa Indonesia...")
            speed_input = gr.Slider(0.7, 1.5, value=1.0, label="Speed")
            btn = gr.Button("🔊 Generate Suara", variant="primary")
        
        with gr.Column():
            wav_output = gr.Audio(label="Hasil WAV")
            mp3_output = gr.Audio(label="Hasil MP3")
            status = gr.Textbox(label="Status")

    btn.click(generate_speech, 
              inputs=[text_input, audio_input, speed_input], 
              outputs=[wav_output, mp3_output, status])

demo.launch(server_name="0.0.0.0", server_port=7860)
