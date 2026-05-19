import os
import streamlit as st
from TTS.api import TTS
import torch
from pydub import AudioSegment
import nltk

# Download data pendukung pemecah kalimat
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')

setup_nltk()
from nltk.tokenize import sent_tokenize

# Setujui lisensi Coqui secara otomatis
os.environ["COQUI_TOS_AGREED"] = "1"

# Mengunci sistem ke mode CPU murni demi kestabilan server free-tier
device = "cpu"

# Menggunakan sistem Caching Streamlit agar model tidak diunduh berulang kali setiap halaman di-refresh
@st.cache_resource
def load_model():
    st.info("Sedang memuat model AI XTTSv2 ke server... Proses awal ini memakan waktu beberapa menit.")
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return model

# Tampilan UI Aplikasi
st.title("🎙️ Iyon Voice Clone - Streamlit Edition")
st.write("Ubah teks panjang menjadi suara kloningan secara online gratis.")

try:
    tts = load_model()
    st.success("Model AI Berhasil Dimuat dan Siap Digunakan!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# Form Input Pengguna
input_text = st.text_area("Masukkan Teks Panjang", placeholder="Ketik kalimat di sini...", height=200)
uploaded_audio = st.file_uploader("Unggah Sampel Suara Target (Format WAV atau MP3, ideal 10-20 detik)", type=["wav", "mp3"])

if st.button("Mulai Clone Suara", type="primary"):
    if not input_text or not uploaded_audio:
        st.warning("Mohon isi teks dan unggah sampel suara terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses kloning suara per kalimat... Mohon tunggu."):
            # Simpan sementara audio unggahan user ke file fisik
            temp_ref_path = "temp_user_reference.wav"
            with open(temp_ref_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            sentences = sent_tokenize(input_text)
            combined_audio = AudioSegment.empty()
            
            temp_folder = "temp_chunks"
            os.makedirs(temp_folder, exist_ok=True)
            
            try:
                progress_bar = st.progress(0)
                for i, sentence in enumerate(sentences):
                    # Update status progres ke pengguna
                    status_text = f"Memproses kalimat {i+1} dari {len(sentences)}..."
                    st.text(status_text)
                    
                    chunk_path = f"{temp_folder}/chunk_{i}.wav"
                    
                    # Eksekusi AI TTS
                    tts.tts_to_file(
                        text=sentence,
                        speaker_wav=temp_ref_path,
                        language="id",
                        file_path=chunk_path
                    )
                    
                    # Gabungkan potongan audio
                    new_chunk = AudioSegment.from_wav(chunk_path)
                    combined_audio += new_chunk
                    
                    # Hapus file sampah kecil
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    
                    # Update bar progress
                    progress_bar.progress((i + 1) / len(sentences))
                
                # Export hasil akhir
                final_output = "hasil_voice_clone.wav"
                combined_audio.export(final_output, format="wav")
                
                st.success("🎉 Kloning Suara Berhasil Selesai!")
                
                # Tampilkan Pemutar Audio di Web
                with open(final_output, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {str(e)}")
            finally:
                # Bersihkan file referensi setelah selesai
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
