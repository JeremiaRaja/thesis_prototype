"""
app_streamlit.py
────────────────
Streamlit frontend untuk IndoBERT Sentiment Analysis — PPKM Tweets
Fitur:
  - Analisis sentimen single tweet (formal & informal)
  - Perbandingan hasil IndoBERT formal vs informal
  - Visualisasi hasil training (grafik & confusion matrix)
  - Tabel perbandingan semua model

Usage:
    streamlit run app_streamlit.py
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndoBERT Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Main header ── */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    color: white;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: rgba(255,255,255,0.8);
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Sentiment result badges ── */
.badge-positif {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}
.badge-negatif {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    color: white;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}
.badge-netral {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a2e;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(102,126,234,0.3);
}
[data-testid="stSidebar"] .stMarkdown {
    color: white;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Space Mono', monospace;
    color: #a78bfa;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(102,126,234,0.3);
}

/* ── Inputs ── */
.stTextArea textarea {
    background: #f8fafc !important;
    border: 1px solid rgba(102,126,234,0.4) !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextArea textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

.stTextArea textarea:focus {
    border: 2px solid #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 6px !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
}

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(102,126,234,0.15);
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #ffffff !important;
    font-size: 0.9rem;
}

/* ── Streamlit info/warning boxes ── */
[data-testid="stNotification"] {
    color: #ffffff !important;
}
.stAlert p, .stAlert div, .stAlert span {
    color: #ffffff !important;
}
/* fix st.info blue box text */
[data-baseweb="notification"] {
    color: #ffffff !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb { background: #667eea; border-radius: 3px; }

/* ===========================
   TAB 2 - Hasil Training
=========================== */

/* Radio Formal / Informal */
.stRadio label,
.stRadio span,
.stRadio p,
div[role="radiogroup"] label,
div[role="radiogroup"] span,
div[role="radiogroup"] p,
[data-baseweb="radio"] label,
[data-baseweb="radio"] span {
    color: #ffffff !important;
}

/* Markdown text (Training Curve & Confusion Matrix) */
.stMarkdown p,
.stMarkdown strong {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODELS_DIR  = "models"
LABEL_MAP   = {0: "negatif", 1: "positif", 2: "netral"}
LABEL_EMOJI = {"positif": "😊", "negatif": "😠", "netral": "😐"}

SLANG_WORDS = set([
    'gak','ga','ngga','nggak','gue','gw','lo','lu','udah','udh',
    'kalo','aja','sih','deh','dong','lah','bgt','banget','emg',
    'emang','yg','dgn','tdk','sdh','blm','msh','krn','gmn',
    'gimana','kayak','kyk','tp','sampe','ampe','wkwk','haha',
    'hehe','nih','guys','gabisa','gausa','gitu','gini','gt',
    'sy','km','mrk','kt','pengen','kesel','capek','males',
    'nanya','bikin','liat','bakal','kok','kan','tuh','plis',
])

import re
def detect_language_type(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    slang_count = sum(1 for t in tokens if t in SLANG_WORDS)
    return "informal" if slang_count >= 2 else "formal"


# ─── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(mode):
    """Load IndoBERT model for given mode (formal/informal)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_path = os.path.join(MODELS_DIR, f"best_IndoBERT_{mode}")
        if not os.path.exists(model_path):
            return None, None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, None


def predict_sentiment(text, tokenizer, model):
    """Run inference and return label + confidence scores."""
    import torch
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred   = int(np.argmax(probs))
    return LABEL_MAP[pred], probs


# ─── Load results helpers ─────────────────────────────────────────────────────
def load_history(mode):
    path = os.path.join(RESULTS_DIR, f"history_IndoBERT_{mode}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_comparison_csv():
    path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def load_report(tag):
    path = os.path.join(RESULTS_DIR, f"report_{tag}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size:2rem;'>📊</div>
        <div style='font-family: Space Mono, monospace; color:#a78bfa; font-size:1rem; font-weight:700;'>IndoBERT</div>
        <div style='color:rgba(255,255,255,0.5); font-size:0.75rem;'>Sentiment Analysis</div>
    </div>
    <hr style='border-color:rgba(102,126,234,0.3);'>
    """, unsafe_allow_html=True)

    st.markdown("**📁 Dataset Info**")
    st.markdown("""
    <div class='info-box'>
        📊 23,644 tweets PPKM<br>
        🗂️ Formal: 21,314<br>
        💬 Informal: 2,330<br>
        🏷️ 3 Kelas: Positif, Negatif, Netral
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**🤖 Model Info**", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        🔬 IndoBERT base-p1<br>
        📐 Max length: 128 tokens<br>
        🔁 Epochs: 5 (early stop)<br>
        📈 LR: 2e-5
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**🎓 Thesis Info**", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Analisis Sentimen Tweet<br>
        PPKM menggunakan IndoBERT<br>
        Data Formal vs Informal
    </div>
    """, unsafe_allow_html=True)


# ─── Main header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>IndoBERT Sentiment Analysis</h1>
    <p>Analisis sentimen tweet PPKM — Formal vs Informal • Tugas Akhir</p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analisis Tweet",
    "📊 Hasil Training",
    "🏆 Perbandingan Model",
    "📋 Classification Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analisis Tweet
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>🔍 Analisis Sentimen Tweet</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Masukkan tweet bahasa Indonesia di bawah ini. Sistem akan otomatis mendeteksi apakah
        bahasa yang digunakan <b>formal</b> atau <b>informal</b>, lalu memilih model IndoBERT
        yang sesuai untuk prediksi sentimen.
    </div>
    """, unsafe_allow_html=True)

    tweet_input = st.text_area(
        "Input Tweet",
        placeholder="Contoh: PPKM bikin ekonomi makin susah, kapan selesainya nih...",
        height=120,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn2:
        analyze_btn = st.button("🔍 Analisis Sekarang", width="stretch")

    if analyze_btn and tweet_input.strip():
        lang_type = detect_language_type(tweet_input)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown(f"""
            <div class='card'>
                <div style='color:rgba(255,255,255,0.5); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Tipe Bahasa Terdeteksi</div>
                <div style='font-family: Space Mono, monospace; font-size:1.4rem; color:#a78bfa; font-weight:700; margin-top:0.5rem;'>
                    {"💬 INFORMAL" if lang_type == "informal" else "📰 FORMAL"}
                </div>
                <div style='color:rgba(255,255,255,0.5); font-size:0.8rem; margin-top:0.5rem;'>
                    {"Menggunakan IndoBERT (informal)" if lang_type == "informal" else "Menggunakan IndoBERT (formal)"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            with st.spinner(f"Memuat IndoBERT ({lang_type})..."):
                tokenizer, model = load_model(lang_type)

            if model is None:
                st.warning(f"""
                ⚠️ Model IndoBERT ({lang_type}) belum tersedia.
                Pastikan training sudah selesai dan folder
                `models/best_IndoBERT_{lang_type}` ada.
                """)
            else:
                with st.spinner("Menganalisis sentimen..."):
                    label, probs = predict_sentiment(tweet_input, tokenizer, model)

                emoji = LABEL_EMOJI[label]
                badge_class = f"badge-{label}"

                st.markdown(f"""
                <div class='card'>
                    <div style='color:rgba(255,255,255,0.5); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Hasil Prediksi</div>
                    <div style='margin: 1rem 0;'>
                        <span class='{badge_class}'>{emoji} {label.upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if model is not None:
            st.markdown("<div class='section-title' style='margin-top:1.5rem;'>📊 Confidence Score</div>",
                        unsafe_allow_html=True)

            labels_order = ["negatif", "positif", "netral"]
            colors       = ["#ef473a", "#38ef7d", "#ffd200"]

            fig, ax = plt.subplots(figsize=(8, 2.5))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            label_to_idx = {v: k for k, v in LABEL_MAP.items()}
            values = [probs[label_to_idx[l]] for l in labels_order]

            bars = ax.barh(labels_order, values, color=colors, height=0.5,
                           edgecolor="none")
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va="center",
                        color="white", fontsize=11, fontweight="bold")

            ax.set_xlim(0, 1.15)
            ax.tick_params(colors="white", labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color((1, 1, 1, 0.2))
            ax.spines["left"].set_color((1, 1, 1, 0.2))
            ax.set_xlabel("Confidence", color="white", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()

    elif analyze_btn and not tweet_input.strip():
        st.warning("⚠️ Masukkan tweet terlebih dahulu!")

    # ── Contoh tweet ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>💡 Contoh Tweet</div>",
                unsafe_allow_html=True)

    examples = {
        "😊 Positif (Formal)":   "Pemerintah telah melonggarkan aturan PPKM untuk mendukung pemulihan ekonomi masyarakat.",
        "😠 Negatif (Informal)": "Aduh PPKM lagi? Udah bosen banget, ekonomi makin susah nih gak ada ujungnya.",
        "😐 Netral (Formal)":    "PPKM level 2 diberlakukan di wilayah Jabodetabek mulai tanggal 1 April 2022.",
    }
    cols = st.columns(3)
    for col, (label_ex, text_ex) in zip(cols, examples.items()):
        with col:
            st.markdown(f"""
            <div style="
                color: #ffffff;
                font-weight: 700;
                font-size: 1rem;
                margin-bottom: 0.6rem;
            ">
                {label_ex}
            </div>
            """, unsafe_allow_html=True)

            st.code(text_ex, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hasil Training
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>📊 Hasil Training IndoBERT</div>",
                unsafe_allow_html=True)

    st.markdown(
        "<div style='color:white;font-weight:600;margin-bottom:0.4rem;'>Pilih Dataset</div>",
        unsafe_allow_html=True
    )

    mode_choice = st.radio(
        "Pilih Dataset",
        ["Formal", "Informal"],
        horizontal=True,
        label_visibility="collapsed",
    )
    mode = mode_choice.lower()

    # Metrics from report
    report_text = load_report(f"IndoBERT_{mode}")
    if report_text:
        acc = f1 = None
        for line in report_text.splitlines():
            if "accuracy" in line.lower():
                try: acc = float(line.split()[-2])
                except: pass
            if "macro avg" in line.lower():
                try: f1 = float(line.split()[-2])
                except: pass

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{acc:.1%}</div>
                <div class='metric-label'>Test Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{f1:.3f}</div>
                <div class='metric-label'>Macro F1-Score</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            hist = load_history(mode)
            best_epoch = int(np.argmax(hist["val_f1"])) + 1 if hist else "-"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>Ep.{best_epoch}</div>
                <div class='metric-label'>Best Epoch</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training curves
    hist = load_history(mode)
    if hist:
        st.markdown(
            f"""
            <h4 style="
                color:white;
                margin-bottom:15px;
                font-weight:700;
            ">
                📈 Training Curve — IndoBERT ({mode_choice})
            </h4>
            """,
            unsafe_allow_html=True
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("none")

        metrics_plot = [
            ("loss", "Loss", "#ef473a"),
            ("acc",  "Accuracy", "#38ef7d"),
            ("f1",   "Macro-F1", "#a78bfa"),
        ]
        for ax, (key, title, color) in zip(axes, metrics_plot):
            ax.set_facecolor("#1a1a2e")
            epochs = range(1, len(hist[f"train_{key}"]) + 1)
            ax.plot(epochs, hist[f"train_{key}"], "o-", color=color,
                    label="Train", linewidth=2, markersize=5)
            ax.plot(epochs, hist[f"val_{key}"], "s--", color="white",
                    label="Val", linewidth=2, markersize=5, alpha=0.7)
            ax.set_title(title, color="white", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", color="white", fontsize=9)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#ffffff")
            ax.legend(fontsize=9, facecolor="#302b63", labelcolor="white")
            ax.grid(True, alpha=0.1)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
    else:
        st.info(f"📂 File `results/history_IndoBERT_{mode}.json` belum ada. Jalankan training terlebih dahulu.")

    # Confusion matrix
    st.markdown(
        """
        <h4 style="
            color:white;
            margin-top:20px;
            margin-bottom:15px;
            font-weight:700;
        ">
            🧩 Confusion Matrix
        </h4>
        """,
        unsafe_allow_html=True
    )
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_IndoBERT_{mode}.png")
    if os.path.exists(cm_path):
        col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
        with col_cm2:
            st.image(cm_path, width="stretch")
    else:
        st.info("📂 Confusion matrix belum tersedia.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Perbandingan Model
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>🏆 Perbandingan Semua Model</div>",
                unsafe_allow_html=True)

    df_comp = load_comparison_csv()
    if df_comp is not None:
        # Highlight best
        best_acc = df_comp["Accuracy"].max()
        best_f1  = df_comp["Macro-F1"].max()

        st.markdown(f"""
        <div style='display:flex; gap:1rem; margin-bottom:1.5rem;'>
            <div class='metric-card' style='flex:1;'>
                <div class='metric-value'>{best_acc:.1%}</div>
                <div class='metric-label'>Best Accuracy</div>
            </div>
            <div class='metric-card' style='flex:1;'>
                <div class='metric-value'>{best_f1:.3f}</div>
                <div class='metric-label'>Best Macro-F1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
        if os.path.exists(chart_path):
            st.image(chart_path, width="stretch")

        # Table
        st.markdown("<br>**Detail Tabel**", unsafe_allow_html=True)
        styled = df_comp.style.format("{:.4f}").highlight_max(
            subset=["Accuracy","Macro-F1"],
            color="#2d1b69"
        )
        st.dataframe(styled, width="stretch")

        # Key findings
        best_model = df_comp["Macro-F1"].idxmax()
        st.markdown(f"""
        <div class='info-box' style='margin-top:1rem;'>
            🏆 <b>Model terbaik:</b> {best_model}<br>
            📈 IndoBERT mengungguli semua baseline klasik di kedua dataset (formal & informal)<br>
            💡 Gap terbesar: IndoBERT vs Logistic Regression pada data formal (+18% Macro-F1)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📂 File `results/model_comparison.csv` belum ada. Jalankan `python -m src.compare` terlebih dahulu.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Classification Report
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>📋 Classification Report Detail</div>",
                unsafe_allow_html=True)

    report_options = {
        "IndoBERT (Formal)":           "IndoBERT_formal",
        "IndoBERT (Informal)":         "IndoBERT_informal",
        "SVM (Formal)":                "svm_formal",
        "SVM (Informal)":              "svm_informal",
        "Naive Bayes (Formal)":        "naive_bayes_formal",
        "Naive Bayes (Informal)":      "naive_bayes_informal",
        "Logistic Regression (Formal)":"logistic_regression_formal",
        "Logistic Regression (Informal)":"logistic_regression_informal",
    }

    selected = st.selectbox("Pilih Model:", list(report_options.keys()))
    report   = load_report(report_options[selected])

    if report:
        lines = report.strip().splitlines()
        rows  = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ("accuracy","macro","weighted"):
                rows.append({
                    "Kelas":     parts[0],
                    "Precision": float(parts[1]),
                    "Recall":    float(parts[2]),
                    "F1-Score":  float(parts[3]),
                    "Support":   int(parts[4]),
                })
            elif "accuracy" in line:
                parts2 = line.split()
                try:
                    rows.append({
                        "Kelas":     "accuracy",
                        "Precision": np.nan,
                        "Recall":    np.nan,
                        "F1-Score":  float(parts2[-2]),
                        "Support":   int(parts2[-1]),
                    })
                except: pass

        if rows:
            df_report = pd.DataFrame(rows).set_index("Kelas")
            formatters = {
                "Precision": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.4f}",
                "Recall": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.4f}",
                "F1-Score": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.4f}",
            }

            st.dataframe(
                df_report.style.format(formatters),
                width="stretch"
            )

        st.markdown("<br>**Raw Report**", unsafe_allow_html=True)
        st.code(report, language=None)
    else:
        st.info(f"📂 Report untuk {selected} belum tersedia.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:rgba(255,255,255,0.3);
     font-size:0.8rem; font-family: Space Mono, monospace;'>
    IndoBERT Sentiment Analysis • Tugas Akhir
</div>
""", unsafe_allow_html=True)