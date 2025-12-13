import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_turnover.joblib")
    model = joblib.load("xgb_turnover_tfidf.joblib")
    df = pd.read_csv("dataset_turnover_746.csv")
    df_rank = pd.read_csv("ranking_kamus_xgb.csv")
    return tfidf, model, df, df_rank

tfidf, model, df, df_rank = load_artifacts()

label_list = ["budaya", "gaji", "karir", "peluang_baru", "wlb"]
label_map = {i: lab for i, lab in enumerate(label_list)}

kamus = {
    "gaji": ["gaji", "underpaid", "bonus", "thr", "tunjangan"],
    "budaya": ["toxic", "bos", "rekan", "micromanagement", "micro management"],
    "karir": ["promosi", "jenjang", "karir", "stuck"],
    "wlb": ["work life balance", "wlb", "lembur", "overtime", "burnout", "overwork"],
    "peluang_baru": ["job offer", "pekerjaan baru", "kerja baru", "pindah kerja", "resign", "tawaran kerja"],
}

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_reason(text: str):
    text_clean = basic_clean(text)
    X_vec = tfidf.transform([text_clean])
    proba = model.predict_proba(X_vec)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_map[pred_idx]
    return pred_label, proba

def detect_keywords(text: str):
    text_low = text.lower()
    hits = []
    for kat, kws in kamus.items():
        for kw in kws:
            if kw in text_low:
                hits.append((kw, kat))
    return hits

st.set_page_config(page_title="Dashboard Turnover Twitter", layout="wide")
st.sidebar.title("Menu")
# Sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih halaman:",
    ("Overview Dataset", "Preprocessing Pipeline", "Klasifikasi Keluhan", "Alasan Utama Turnover")
)

elif page == "Preprocessing Pipeline":
    st.title("Preprocessing Data Crawling")
    uploaded = st.file_uploader("Upload file CSV mentah", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write("Preview data mentah:")
        st.dataframe(df_raw.head())

        if st.button("Jalankan preprocessing"):
            # sementara dummy dulu
            st.warning("Fungsi run_preprocessing() belum diimplementasikan.")
            # nanti di sini kita panggil run_preprocessing(df_raw)

elif page == "Alasan Utama Turnover":
    st.title("Alasan Utama Turnover (Feature Importance)")
    top_n = st.slider("Tampilkan berapa kata kunci teratas?", 5, 30, 15)
    df_top = df_rank.head(top_n)
    st.dataframe(df_top[["rank", "keyword", "importance"]])
    st.bar_chart(df_top.set_index("keyword")["importance"])
if page == "Overview Dataset":
    st.title("Overview Dataset Turnover Twitter")
    st.metric("Jumlah tweet turnover (setelah filter)", len(df))
    st.subheader("Distribusi Kategori Alasan")
    cat_counts = df["kategori"].value_counts().reindex(label_list)
    st.bar_chart(cat_counts)
        st.subheader("Proporsi Kategori Alasan (Pie Chart)")
    df_pie = cat_counts.reset_index()
    df_pie.columns = ["kategori", "jumlah"]
    st.plotly_chart(
        px.pie(df_pie, names="kategori", values="jumlah", hole=0.3),
        use_container_width=True
    )

elif page == "Klasifikasi Keluhan":
    st.title("Klasifikasi Keluhan Kerja")
    text_input = st.text_area("Teks keluhan / tweet", height=150)
    if st.button("Analisis"):
        if not text_input.strip():
            st.warning("Tolong isi teks terlebih dahulu.")
        else:
            pred_label, proba = predict_reason(text_input)
            st.subheader(f"Hasil prediksi kategori: **{pred_label}**")
            proba_dict = {label_map[i]: float(p) for i, p in enumerate(proba)}
            st.bar_chart(proba_dict)
            hits = detect_keywords(text_input)
            if hits:
                st.write("Kata kunci kamus yang terdeteksi:")
                for kw, kat in hits:
                    st.write(f"- `{kw}` → **{kat}**")
            else:
                st.write("Tidak ada kata kunci kamus yang terdeteksi.")

elif page == "Alasan Utama Turnover":
    st.title("Alasan Utama Turnover (Feature Importance)")
    top_n = st.slider("Tampilkan berapa kata kunci teratas?", 5, 30, 15)
    df_top = df_rank.head(top_n)
    st.dataframe(df_top[["rank", "keyword", "importance"]])
    st.bar_chart(df_top.set_index("keyword")["importance"])
