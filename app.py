import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# --- Load model dan tokenizer ---
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load("model_indoBERT_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Fungsi Prediksi ---
def predict_hoax(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    prob_real = probs[0][0].item()
    prob_hoax = probs[0][1].item()
    label = "Hoaks" if prob_hoax > prob_real else "Real"
    return label, prob_hoax * 100, prob_real * 100

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Berita Hoaks",
    page_icon="favicon.png",
    layout="centered"
)

# --- Judul Halaman ---
st.markdown("<h1 style='text-align;'>📰 Deteksi Berita Hoaks Bahasa Indonesia</h1>", unsafe_allow_html=True)

# --- Input ---
input_text = st.text_area(" ", height=180, label_visibility="collapsed")

# --- Tombol Prediksi rata kanan ---
col_spacer, col_button = st.columns([5, 1])
with col_button:
    predict = st.button("🔍 Prediksi")

# --- Hasil Prediksi ---
if predict:
    if input_text.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    elif len(input_text.split()) < 5:
        st.warning("⚠️ Masukkan minimal 5 kata untuk akurasi prediksi yang lebih baik.")
    else:
        label, prob_hoax, prob_real = predict_hoax(input_text)

        warna = "red" if label == "Hoaks" else "green"
        ikon = "❌" if label == "Hoaks" else "✅"

        st.markdown(f"""
            <div style='text-align:center; padding: 10px; background-color: #f9f9f9; border-radius: 10px;'>
                <h2 style='color:{warna};'>{ikon} Hasil Prediksi: {label}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob_hoax if label == "Hoaks" else prob_real))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div style='text-align:center; font-size:22px; font-weight:bold;'>Probabilitas Hoaks</div>
                <div style='text-align:center; font-size:28px; color:red;'>{prob_hoax:.2f}%</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style='text-align:center; font-size:22px; font-weight:bold;'>Probabilitas Real</div>
                <div style='text-align:center; font-size:28px; color:green;'>{prob_real:.2f}%</div>
            """, unsafe_allow_html=True)

# --- Footer: Ikon Kontak + Copyright ---
st.markdown("<br><br>", unsafe_allow_html=True)

linkedin_url = "https://www.linkedin.com/in/dyan-aqlima-febriyanti/"
email_address = "dyanaqlima323@gmail.com"

st.markdown(f"""
    <div style='text-align: center; font-size: 14px;'>
        <a href="{linkedin_url}" target="_blank" style="margin-right: 15px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"/>
        </a>
        <a href="mailto:{email_address}" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="25"/>
        </a>
    </div>
    <div style='text-align: center; font-size: 13px; margin-top: 5px;'>
        © 2025 | Dyan Aqlima Febriyanti | All rights reserved
    </div>
""", unsafe_allow_html=True)
