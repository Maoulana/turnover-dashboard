import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import altair as alt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# LOAD ARTEFAK MODEL
# =========================
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_turnover.joblib")
    model = joblib.load("xgb_turnover_tfidf.joblib")
    df = pd.read_csv("dataset_turnover_900.csv")
    df_rank = pd.read_csv("ranking_kamus_xgb.csv")          # FI XGBoost
    df_shap_k = pd.read_csv("shap_kamus_importance.csv")    # SHAP kamus
    return tfidf, model, df, df_rank, df_shap_k

tfidf, model, df, df_rank, df_shap_k = load_artifacts()

label_list = ["budaya", "gaji", "karir", "peluang_baru", "wlb"]
label_map = {i: lab for i, lab in enumerate(label_list)}

kamus_deteksi = {
    "gaji": [
        "gaji kecil", "gaji rendah", "underpaid", "tunjangan", "bonus tahunan",
        "insentif", "thr"
    ],
    "budaya": [
        "rekan bossy", "bos toxic", "atasan toxic", "rekan kerja toxic",
        "micro management", "micromanagement", "manajemen buruk"
    ],
    "karir": [
        "karir stagnan", "promosi jabatan", "jenjang karir", "karir tidak berkembang",
        "jabatan stuck", "karir stuck"
    ],
    "wlb": [
        "work life balance", "lembur", "overtime", "wlb", "burnout",
        "overwork", "masuk terus"
    ],
    "peluang_baru": [
        "pekerjaan baru", "job offer", "tawaran kerja", "offer kerja",
        "diterima kerja baru", "cari kerja baru"
    ],
}

# =========================
# FUNGSI CLEANING & PREDIKSI
# =========================
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
    for kat, kws in kamus_deteksi.items():
        for kw in kws:
            if kw in text_low:
                hits.append((kw, kat))
    return hits

# =========================
# PREPROCESSING PIPELINE
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

turnover_keywords = [
    r"\bresign\b",
    r"\bpengen resign\b",
    r"\bpengen cabut\b",
    r"\bpingin resign\b",
    r"\bpingin cabut\b",
    r"\bga betah kerja\b",
    r"\bgak betah kerja\b",
    r"\bkeluar kerja\b",
    r"\bberhenti kerja\b",
    r"\bberhenti\b",
    r"\bpindah kerja\b",
    r"\bcabut\b",
    r"\bmutasi\b",
    r"\bjob offer\b",
    r"\bjoboffer\b",
    r"\bditerima kerja baru\b",
    r"\btawaran kerja\b",
    r"\boffer kerja\b",
]
pattern_turnover = re.compile("|".join(turnover_keywords), flags=re.IGNORECASE)

neg_gaji_keywords = [
    r"\bgaji kecil\b",
    r"\bgaji rendah\b",
    r"\bunderpaid\b",
    r"\bgaji ga seberapa\b",
    r"\bgaji gak seberapa\b",
    r"\bgaji ga sebanding\b",
    r"\bgaji gak sebanding\b",
    r"\bgaji pas-pasan\b",
    r"\bthr telat\b",
    r"\bthr gak ada\b",
    r"\bthr ga ada\b",
    r"\bthr pelit\b",
    r"\bbonus pelit\b",
]
pattern_neg_gaji = re.compile("|".join(neg_gaji_keywords), flags=re.IGNORECASE)

neg_budaya_keywords = [
    r"\bbos toxic\b",
    r"\batasan toxic\b",
    r"\brekan kerja toxic\b",
    r"\bteman kerja toxic\b",
    r"\btempat kerja toxic\b",
    r"\btoxic workplace\b",
    r"\bmicro management\b",
    r"\bmicromanagement\b",
    r"\bmanajemen buruk\b",
    r"\bdi-bully\b",
    r"\bdibully\b",
    r"\bdirundung\b",
]
pattern_neg_budaya = re.compile("|".join(neg_budaya_keywords), flags=re.IGNORECASE)

neg_karir_keywords = [
    r"\bkarir stagnan\b",
    r"\bkarir mentok\b",
    r"\bkarir stuck\b",
    r"\bkarir tidak berkembang\b",
    r"\bkarir ga berkembang\b",
    r"\bkarir gak berkembang\b",
    r"\bjenjang karir\b",
    r"\bga naik pangkat\b",
    r"\bgak naik pangkat\b",
    r"\bpromosi mandek\b",
    r"\bjabatan stuck\b",
]
pattern_neg_karir = re.compile("|".join(neg_karir_keywords), flags=re.IGNORECASE)

neg_wlb_keywords = [
    r"\bwork life balance\b",
    r"\bwlb\b",
    r"\blembur terus\b",
    r"\blembur mulu\b",
    r"\bovertime terus\b",
    r"\bkerja terus\b",
    r"\bkerja dari pagi sampe malem\b",
    r"\bkerja dari pagi sampai malam\b",
    r"\bkerja ga ada libur\b",
    r"\bkerja gak ada libur\b",
    r"\bburnout\b",
    r"\boverwork\b",
    r"\bcapek kerja\b",
    r"\bcapek banget kerja\b",
]
pattern_neg_wlb = re.compile("|".join(neg_wlb_keywords), flags=re.IGNORECASE)

def stem_indonesian(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return stemmer.stem(text)

def is_turnover_related(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if pattern_turnover.search(text):
        return True
    if (
        pattern_neg_gaji.search(text)
        or pattern_neg_budaya.search(text)
        or pattern_neg_karir.search(text)
        or pattern_neg_wlb.search(text)
    ):
        return True
    return False

kamus_pre = {
    "gaji": ["gaji kecil", "gaji rendah", "underpaid", "tunjangan", "bonus", "bonus tahunan", "insentif", "thr"],
    "budaya": ["rekan bossy", "bos toxic", "atasan toxic", "rekan kerja toxic", "micro management", "micromanagement", "manajemen buruk"],
    "karir": ["karir stagnan", "promosi jabatan", "jenjang karir", "karir tidak berkembang", "jabatan stuck", "karir stuck"],
    "wlb": ["work life balance", "lembur", "overtime", "wlb", "burnout", "overwork", "masuk terus"],
    "peluang_baru": ["pekerjaan baru", "job offer", "tawaran kerja", "offer kerja", "diterima kerja baru", "cari kerja baru"],
}

def label_kategori_pre(text: str) -> str:
    text_low = text.lower() if isinstance(text, str) else ""
    for kat, kws in kamus_pre.items():
        if any(kw in text_low for kw in kws):
            return kat
    return "lainnya"

def run_preprocessing(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_work = df_raw.copy()
    if "full_text" not in df_work.columns:
        raise ValueError("Kolom 'full_text' tidak ditemukan di file upload.")
    df_work["text_clean_basic"] = df_work["full_text"].apply(basic_clean)
    df_work["text_stem"] = df_work["text_clean_basic"].apply(stem_indonesian)
    df_work["is_turnover_related"] = df_work["text_stem"].apply(is_turnover_related)
    df_turnover = df_work[df_work["is_turnover_related"]].reset_index(drop=True)
    df_turnover["kategori"] = df_turnover["text_stem"].apply(label_kategori_pre)
    return df_turnover

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Dashboard Turnover Twitter", layout="wide")

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih halaman:",
    (
        "Overview Dataset",
        "Preprocessing Pipeline",
        "Klasifikasi Keluhan",
        "Alasan Utama Turnover",
        "Analisis SHAP Kamus",
    ),
)

# ---------- Overview ----------
if page == "Overview Dataset":
    st.title("Overview Dataset Turnover Twitter")
    st.metric("Jumlah tweet turnover/keluhan (setelah filter)", len(df))

    st.subheader("Distribusi Kategori Alasan (Bar Chart)")
    cat_counts = df["kategori"].value_counts().reindex(label_list)
    st.bar_chart(cat_counts)

    st.subheader("Proporsi Kategori Alasan (Pie Chart)")
    df_pie = cat_counts.reset_index()
    df_pie.columns = ["kategori", "jumlah"]
    fig_pie = px.pie(df_pie, names="kategori", values="jumlah", hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------- Preprocessing ----------
elif page == "Preprocessing Pipeline":
    st.title("Preprocessing Data Crawling")
    uploaded = st.file_uploader("Upload file CSV mentah (kolom 'full_text')", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write("Preview data mentah:")
        st.dataframe(df_raw.head())

        if st.button("Jalankan preprocessing"):
            try:
                df_clean = run_preprocessing(df_raw)
                st.success(
                    f"Preprocessing selesai. Jumlah tweet setelah filter turnover/keluhan: {len(df_clean)}"
                )
                st.write("Preview hasil preprocessing:")
                st.dataframe(df_clean.head())

                csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV hasil preprocessing",
                    data=csv_bytes,
                    file_name="hasil_preprocessing_turnover.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Terjadi error saat preprocessing: {e}")

# ---------- Klasifikasi ----------
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

# ---------- Feature Importance (XGBoost FI) ----------
elif page == "Alasan Utama Turnover":
    st.title("Alasan Utama Turnover (Feature Importance XGBoost)")
    max_n = len(df_rank)
    top_n = st.slider("Tampilkan berapa kata kunci teratas?", 5, max_n, min(15, max_n))
    df_top = df_rank.head(top_n)
    st.dataframe(df_top[["rank", "keyword", "importance"]])
    st.bar_chart(df_top.set_index("keyword")["importance"])

# ---------- SHAP Kamus ----------
elif page == "Analisis SHAP Kamus":
    st.title("Analisis Alasan Resign (SHAP)")

    max_n = len(df_shap_k)
    top_n = st.slider("Tampilkan berapa alasan teratas?", 5, min(25, max_n), 10)

    df_top = df_shap_k.head(top_n)

    chart = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap", title="Rata-rata |SHAP|"),
            y=alt.Y("feature", sort="-x", title="Keyword / Alasan"),
            tooltip=["feature", "mean_abs_shap"],
        )
    )

    st.altair_chart(chart, use_container_width=True)
