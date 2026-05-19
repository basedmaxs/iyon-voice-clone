import os
import streamlit as st
from TTS.api import TTS
import torch
from pydub import AudioSegment
import nltk

# 1. Inisialisasi Resource (NLTK & OS Environment)
@st.cache_resource
def init_resources():
    nltk.download('punkt')
    os.environ["COQUI_TOS_AGREED"] = "1"
    return True

init_resources()
from nltk.tokenize import sent_tokenize

# 2. Muat Model XTTSv2 (Dibuat Cache agar tidak download ulang terus)
@st.cache_resource
def load_xtts_model():
    # Menggunakan gpu=False agar aman di server gratis Streamlit
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    return model

# Panggil fungsi muat model
try:
    tts = load_xtts_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Tampilan Antarmuka (UI) Streamlit
st.set_page_config(page_title="Iyon Voice Clone", page_icon="🎙️")
st.title("🎙️ Iyon AI Voice Clone")
st.write("Ubah teks menjadi suara kloningan menggunakan CPU Server.")

# Area Input
input_text = st.text_area("Masukkan Teks", placeholder="Ketik kalimat di sini...", height=150)
uploaded_audio = st.file_uploader("Unggah Sampel Suara (WAV/MP3)", type=["wav", "mp3"])

# Tombol Proses
if st.button("Mulai Clone Suara", type="primary"):
    if not input_text or not uploaded_audio:
        st.warning("Teks dan Sampel Suara tidak boleh kosong!")
    else:
        with st.spinner("Sedang memproses... Harap tunggu (ini memakan waktu di CPU)."):
            # Simpan referensi suara sementara
            temp_ref = "temp_reference.wav"
            with open(temp_ref, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            try:
                # Pecah teks menjadi kalimat
                sentences = sent_tokenize(input_text)
                combined_audio = AudioSegment.empty()
                
                # Buat folder sementara untuk potongan suara
                os.makedirs("chunks", exist_ok=True)
                
                # Progress bar untuk user
                progress_bar = st.progress(0)
                
                for i, sent in enumerate(sentences):
                    st.text(f"Memproses kalimat {i+1} dari {len(sentences)}...")
                    chunk_path = f"chunks/chunk_{i}.wav"
                    
                    # Proses TTS
                    tts.tts_to_file(
                        text=sent,
                        speaker_wav=temp_ref,
                        language="id",
                        file_path=chunk_path
                    )
                    
                    # Gabungkan Audio
                    new_chunk = AudioSegment.from_wav(chunk_path)
                    combined_audio += new_chunk
                    
                    # Hapus potongan agar hemat disk
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(sentences))
                
                # Simpan hasil akhir
                output_final = "hasil_clone.wav"
                combined_audio.export(output_final, format="wav")
                
                st.success("Selesai!")
                st.audio(output_final)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            finally:
                # Bersihkan file referensi
                if os.path.exists(temp_ref):
                    os.remove(temp_ref)
