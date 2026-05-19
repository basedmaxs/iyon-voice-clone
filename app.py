@st.cache_resource
def load_model():
    st.info("Sedang memuat model... Harap bersabar, ini memakan waktu sekitar 3-5 menit.")
    # Menambahkan parameter 'use_deepspeed=False' agar lebih hemat RAM
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    model.to(device)
    return model
