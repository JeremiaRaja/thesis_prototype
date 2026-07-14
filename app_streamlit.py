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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
LABEL_MAPS = {
    "formal":   {0: "netral", 1: "positif", 2: "negatif"},
    "informal": {0: "netral", 1: "positif", 2: "negatif"},
}
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


def predict_sentiment(text, tokenizer, model, label_map):
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
    return label_map[pred], probs


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

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,26,46,0.8)",
    font=dict(color="white", family="DM Sans"),
    margin=dict(l=40, r=40, t=50, b=40),
)

CM_VALUES = {
    "IndoBERT_formal":   [[174, 50,   8],
                          [68,  2409, 62],
                          [22,  83,  322]],
    "IndoBERT_informal": [[47,  4,   11],
                          [13,  91,  13],
                          [17,  5,  149]],
}

def plot_confusion_matrix_plotly(tag, mode):
    if tag not in CM_VALUES:
        return None
    cm     = CM_VALUES[tag]
    labels = ["negatif", "positif", "netral"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        showscale=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Confusion Matrix — IndoBERT ({mode.capitalize()})",
                   font=dict(color="white", size=14)),
        xaxis=dict(title="Predicted", title_font=dict(color="white"),
                   tickfont=dict(color="white")),
        yaxis=dict(title="Actual", title_font=dict(color="white"),
                   tickfont=dict(color="white"), autorange="reversed"),
        height=420,
    )
    return fig


def plot_model_comparison_plotly(df):
    models    = df.index.tolist()
    acc       = df["Accuracy"].tolist()
    f1        = df["Macro-F1"].tolist()
    acc_colors = ["#667eea" if "IndoBERT" in m else "#4a5568" for m in models]
    f1_colors  = ["#a78bfa" if "IndoBERT" in m else "#6b7280" for m in models]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=models, y=acc, marker_color=acc_colors,
        text=[f"{v:.3f}" for v in acc], textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.add_trace(go.Bar(
        name="Macro-F1", x=models, y=f1, marker_color=f1_colors,
        text=[f"{v:.3f}" for v in f1], textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="IndoBERT Formal vs Informal — Full Model Comparison",
                   font=dict(color="white", size=15)),
        xaxis=dict(tickangle=-20, tickfont=dict(color="white", size=10),
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[0, 1.15], tickfont=dict(color="white"),
                   gridcolor="rgba(255,255,255,0.1)", title="Score",
                   title_font=dict(color="white")),
        legend=dict(bgcolor="rgba(30,30,60,0.8)", bordercolor="#667eea",
                    borderwidth=1, font=dict(color="white")),
        height=500,
    )
    return fig


def load_kfold_data():
    """Load all K-Fold result files."""
    results = {}
    base = "results"
    files = {
        "combined":       "combined_fold_results.csv",
        "mean_std":       "mean_std_summary.csv",
        "significance":   "significance_test_results.csv",
        "all_folds_formal":   "all_fold_metrics_formal.csv",
        "all_folds_informal": "all_fold_metrics_informal.csv",
    }
    for key, fname in files.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            results[key] = pd.read_csv(path)
    return results


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Analisis Tweet",
    "📊 Hasil Training",
    "🏆 Perbandingan Model",
    "📋 Classification Report",
    "🔁 K-Fold Validation",
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
                    label_map = LABEL_MAPS[lang_type]
                    label, probs = predict_sentiment(tweet_input, tokenizer, model, label_map)
                
                # st.write("Model label config:", model.config.id2label)
                # st.write("Raw probs:", probs)
                # st.write("Pred index:", int(np.argmax(probs)))

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

            label_to_idx = {v: k for k, v in label_map.items()}
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
    fig_cm = plot_confusion_matrix_plotly(f"IndoBERT_{mode}", mode)
    if fig_cm:
        col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
        with col_cm2:
            st.plotly_chart(fig_cm, use_container_width=True)
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

        # Bar chart — Plotly interaktif
        df_plot = df_comp[["Accuracy","Macro-F1"]].dropna()
        st.plotly_chart(plot_model_comparison_plotly(df_plot), use_container_width=True)

        # Table
        st.markdown("<br>**Detail Tabel**", unsafe_allow_html=True)
        df_display = df_comp.copy()
        for col in ["Accuracy", "Macro-F1", "Weighted-F1"]:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors="coerce")
        styled = df_display.style.format("{:.4f}", na_rep="-").highlight_max(
            subset=[c for c in ["Accuracy","Macro-F1"] if c in df_display.columns],
            color="#2d1b69"
        )
        st.dataframe(styled, use_container_width=True)

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




# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — K-Fold Cross Validation
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>🔁 K-Fold Cross Validation (5-Fold)</div>",
                unsafe_allow_html=True)

    kfold_data = load_kfold_data()

    if not kfold_data:
        st.info("📂 File hasil K-Fold belum tersedia di folder results/.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("### 📊 Ringkasan Hasil K-Fold")

        formal_summary   = {"accuracy": 0.9509, "macro_precision": 0.8653,
                            "macro_recall": 0.8555, "macro_f1": 0.8601}
        informal_summary = {"accuracy": 0.8784, "macro_precision": 0.8588,
                            "macro_recall": 0.8634, "macro_f1": 0.8606}

        if "mean_std" in kfold_data:
            df_ms = kfold_data["mean_std"]
            formal_row   = df_ms[df_ms["dataset"] == "formal"].iloc[0]
            informal_row = df_ms[df_ms["dataset"] == "informal"].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class='card'>
                <div style='font-family:Space Mono,monospace; color:#667eea;
                     font-size:1rem; font-weight:700; margin-bottom:1rem;'>
                    📰 FORMAL
                </div>""", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{formal_row['accuracy_mean_std'] if 'mean_std' in kfold_data else '0.9509 ± 0.0025'}</div>
                    <div class='metric-label'>Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{formal_row['macro_f1_mean_std'] if 'mean_std' in kfold_data else '0.8601 ± 0.0071'}</div>
                    <div class='metric-label'>Macro F1</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class='card'>
                <div style='font-family:Space Mono,monospace; color:#a78bfa;
                     font-size:1rem; font-weight:700; margin-bottom:1rem;'>
                    💬 INFORMAL
                </div>""", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{informal_row['accuracy_mean_std'] if 'mean_std' in kfold_data else '0.8784 ± 0.0099'}</div>
                    <div class='metric-label'>Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{informal_row['macro_f1_mean_std'] if 'mean_std' in kfold_data else '0.8606 ± 0.0100'}</div>
                    <div class='metric-label'>Macro F1</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Per-fold line chart ───────────────────────────────────────────────
        st.markdown("### 📈 Hasil Per Fold")
        metric_choice = st.selectbox(
            "Pilih Metrik:",
            ["accuracy", "macro_f1", "macro_precision", "macro_recall"],
            format_func=lambda x: x.replace("_", " ").title()
        )

        if "combined" in kfold_data:
            df_combined = kfold_data["combined"]
            df_formal   = df_combined[df_combined["dataset"] == "formal"].sort_values("fold")
            df_informal = df_combined[df_combined["dataset"] == "informal"].sort_values("fold")

            fig_fold = go.Figure()
            fig_fold.add_trace(go.Scatter(
                x=df_formal["fold"], y=df_formal[metric_choice],
                mode="lines+markers", name="Formal",
                line=dict(color="#667eea", width=2),
                marker=dict(size=8, symbol="circle"),
            ))
            fig_fold.add_trace(go.Scatter(
                x=df_informal["fold"], y=df_informal[metric_choice],
                mode="lines+markers", name="Informal",
                line=dict(color="#a78bfa", width=2, dash="dash"),
                marker=dict(size=8, symbol="square"),
            ))

            # Mean lines
            fig_fold.add_hline(
                y=df_formal[metric_choice].mean(),
                line_dash="dot", line_color="#667eea", opacity=0.5,
                annotation_text=f"Formal mean: {df_formal[metric_choice].mean():.4f}",
                annotation_font_color="#667eea",
            )
            fig_fold.add_hline(
                y=df_informal[metric_choice].mean(),
                line_dash="dot", line_color="#a78bfa", opacity=0.5,
                annotation_text=f"Informal mean: {df_informal[metric_choice].mean():.4f}",
                annotation_font_color="#a78bfa",
            )

            fig_fold.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f"{metric_choice.replace('_',' ').title()} per Fold — Formal vs Informal",
                    font=dict(color="white", size=14)
                ),
                xaxis=dict(title="Fold", tickvals=[1,2,3,4,5],
                           tickfont=dict(color="white"),
                           gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title=metric_choice.replace("_"," ").title(),
                           tickfont=dict(color="white"),
                           gridcolor="rgba(255,255,255,0.1)",
                           title_font=dict(color="white")),
                legend=dict(bgcolor="rgba(30,30,60,0.8)", bordercolor="#667eea",
                            borderwidth=1, font=dict(color="white")),
                height=420,
            )
            st.plotly_chart(fig_fold, use_container_width=True)

        # ── Detail table per fold ─────────────────────────────────────────────
        st.markdown("### 📋 Detail Hasil Tiap Fold")
        dataset_choice = st.radio("Dataset:", ["Formal", "Informal"],
                                  horizontal=True, key="kfold_dataset")

        if "combined" in kfold_data:
            df_detail = kfold_data["combined"]
            df_detail = df_detail[
                df_detail["dataset"] == dataset_choice.lower()
            ][["fold","accuracy","macro_precision","macro_recall","macro_f1","eval_loss"]].copy()
            df_detail.columns = ["Fold","Accuracy","Precision","Recall","Macro-F1","Eval Loss"]
            df_detail = df_detail.set_index("Fold")

            st.dataframe(
                df_detail.style.format("{:.4f}").highlight_max(
                    subset=["Accuracy","Macro-F1"], color="#2d1b69"
                ).highlight_min(
                    subset=["Eval Loss"], color="#1a3a2a"
                ),
                use_container_width=True
            )

        # ── Significance test ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 Uji Signifikansi (Welch t-test & Mann-Whitney)")

        st.markdown("""<div class='info-box'>
            Uji signifikansi dilakukan untuk mengetahui apakah perbedaan performa
            IndoBERT pada data <b>formal</b> vs <b>informal</b> signifikan secara statistik
            (α = 0.05).
        </div>""", unsafe_allow_html=True)

        sig_data = {
            "Metrik": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
            "Formal Mean": [0.9509, 0.8653, 0.8555, 0.8601],
            "Informal Mean": [0.8784, 0.8588, 0.8634, 0.8606],
            "p-value (Welch)": [0.000039, 0.360174, 0.269822, 0.927569],
            "p-value (Mann-Whitney)": [0.011925, 0.222222, 0.309524, 1.000000],
            "Signifikan?": ["✅ Ya", "❌ Tidak", "❌ Tidak", "❌ Tidak"],
        }

        if "significance" in kfold_data:
            df_sig = kfold_data["significance"]
            sig_data["p-value (Welch)"]        = df_sig["welch_t_p_value"].tolist()
            sig_data["p-value (Mann-Whitney)"] = df_sig["mann_whitney_p_value"].tolist()
            sig_data["Signifikan?"]            = [
                "✅ Ya" if v else "❌ Tidak"
                for v in df_sig["significant_welch_0.05"].tolist()
            ]

        df_sig_display = pd.DataFrame(sig_data).set_index("Metrik")
        st.dataframe(
            df_sig_display.style.format({
                "Formal Mean": "{:.4f}",
                "Informal Mean": "{:.4f}",
                "p-value (Welch)": "{:.6f}",
                "p-value (Mann-Whitney)": "{:.6f}",
            }),
            use_container_width=True
        )

        # Interpretation
        st.markdown("""<div class='info-box' style='margin-top:1rem;'>
            📌 <b>Interpretasi:</b><br>
            • <b>Accuracy</b>: Terdapat perbedaan <b>signifikan</b> (p &lt; 0.05) antara formal dan informal —
            model lebih akurat pada data formal (95.09%) vs informal (87.84%)<br>
            • <b>Macro F1</b>: <b>Tidak signifikan</b> (p = 0.928) — kemampuan model dalam mendeteksi
            semua kelas secara seimbang hampir sama di kedua dataset<br>
            • Kesimpulan: IndoBERT efektif untuk kedua jenis bahasa, dengan keunggulan accuracy pada data formal
        </div>""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:rgba(255,255,255,0.3);
     font-size:0.8rem; font-family: Space Mono, monospace;'>
    IndoBERT Sentiment Analysis • Tugas Akhir
</div>
""", unsafe_allow_html=True)