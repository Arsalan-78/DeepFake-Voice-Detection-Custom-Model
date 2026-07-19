import html
import json
import os
import pickle
import time

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Sentinel · Voice Authenticity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = "results"


# ===========================================================================
# Design system (single source of truth)
# ===========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0a0e1a;
  --surface: rgba(17, 24, 39, 0.72);
  --surface-2: rgba(2, 6, 23, 0.5);
  --border: rgba(148, 163, 184, 0.13);
  --border-2: rgba(148, 163, 184, 0.26);
  --text: #e8eef7;
  --muted: #93a1b5;
  --faint: #64748b;
  --accent: #6ee7d3;
  --accent-2: #56b8f0;
  --ink: #072022;
  --panel: #111a2e;
  --ok: #34d399;
  --bad: #fb7185;
  --radius: 16px;
  --radius-sm: 11px;
  --shadow: 0 18px 50px rgba(0,0,0,0.30);
  --gap: 1.2rem;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1000px 560px at 8% -10%, rgba(110,231,211,0.09), transparent 60%),
    radial-gradient(760px 520px at 96% 4%, rgba(86,184,240,0.07), transparent 60%),
    var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
* { font-family: 'Inter', sans-serif; }

.block-container { padding: 2.1rem 2.4rem 3rem; max-width: 1160px; }

/* Hide default chrome for a cleaner product feel */
#MainMenu, header [data-testid="stToolbar"], footer { visibility: hidden; }

[data-testid="stSidebar"] {
  background: rgba(2, 6, 23, 0.96);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }

h1, h2, h3, h4 { letter-spacing: -0.025em; color: var(--text); margin: 0; }
p, li, span, label { color: inherit; }

/* ---- Typography scale ---- */
.h-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.035em; }
.h-sub   { color: var(--muted); font-size: 0.98rem; line-height: 1.6; margin-top: 0.35rem; }
.eyebrow { color: var(--muted); font-size: 0.71rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }

/* ---- Layout primitives ---- */
.shell { display: flex; flex-direction: column; gap: var(--gap); }
.page-header { padding: 0.2rem 0.2rem 0.4rem; }

.card {
  padding: 1.35rem 1.45rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  backdrop-filter: blur(14px);
}
.card.tight { padding: 1.1rem 1.25rem; }
.card h3, .card-h { font-size: 1.06rem; font-weight: 700; margin: 0.35rem 0 0.3rem; color: var(--text); }
.card p  { color: var(--muted); font-size: 0.9rem; line-height: 1.6; margin: 0; }
.card-copy { color: var(--muted); font-size: 0.9rem; line-height: 1.6; margin: 0 0 1rem; }
.steps { color: var(--muted); font-size: 0.9rem; line-height: 1.9; }
.support-note { color: var(--faint); font-size: 0.8rem; margin-top: 0.4rem; }

/* Bordered Streamlit containers rendered as design-system cards.
   This lets live widgets (uploader, tabs, images) live INSIDE a card. */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface);
  backdrop-filter: blur(14px);
  padding: 1.25rem 1.35rem;
}

/* ---- Brand ---- */
.brand { display: flex; align-items: center; gap: 0.7rem; }
.brand-mark {
  width: 40px; height: 40px; border-radius: 12px; flex: none;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  display: flex; align-items: center; justify-content: center;
  color: var(--ink); font-weight: 800; box-shadow: 0 10px 28px rgba(110,231,211,0.24);
}
.brand-text strong { display: block; font-size: 0.98rem; font-weight: 700; color: #fff; }
.brand-text span { display: block; font-size: 0.76rem; color: var(--muted); }

/* ---- Pills / badges ---- */
.pill {
  display: inline-flex; align-items: center; gap: 0.45rem;
  padding: 0.32rem 0.72rem; border-radius: 999px;
  font-size: 0.76rem; font-weight: 700;
  border: 1px solid var(--border-2); background: var(--surface-2); color: var(--muted);
}
.pill .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--muted); }
.pill.ok  { color: #b9f5dd; border-color: rgba(52,211,153,0.35); background: rgba(52,211,153,0.10); }
.pill.ok .dot  { background: var(--ok); box-shadow: 0 0 0 3px rgba(52,211,153,0.16); }
.pill.bad { color: #fecdd3; border-color: rgba(251,113,133,0.36); background: rgba(251,113,133,0.10); }
.pill.bad .dot { background: var(--bad); }

/* ---- Metric ---- */
.metric { padding: 1rem 1.1rem; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); height: 100%; }
.metric span { display: block; color: var(--muted); font-size: 0.71rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.metric strong { display: block; margin-top: 0.4rem; color: #fff; font-size: 1.34rem; font-weight: 700; }
.metric small { display: block; margin-top: 0.2rem; color: var(--faint); font-size: 0.74rem; font-weight: 500; }

/* ---- Confidence bars ---- */
.conf-row { margin: 0.8rem 0; }
.conf-row:first-child { margin-top: 0.5rem; }
.conf-top { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.84rem; font-weight: 600; margin-bottom: 0.4rem; }
.bar-track { width: 100%; height: 10px; border-radius: 999px; background: rgba(148,163,184,0.16); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transition: width .45s ease; }
.bar-fill.fake { background: linear-gradient(90deg, #fb7185, #f43f5e); }
.bar-fill.real { background: linear-gradient(90deg, #34d399, #6ee7d3); }

/* ---- Result ---- */
.result { position: relative; overflow: hidden; padding: 1.6rem; border-radius: var(--radius); border: 1px solid var(--border-2); background: var(--surface); }
.result.real { border-color: rgba(52,211,153,0.40); background: linear-gradient(135deg, rgba(52,211,153,0.10), var(--surface)); }
.result.fake { border-color: rgba(251,113,133,0.42); background: linear-gradient(135deg, rgba(251,113,133,0.10), var(--surface)); }
.result .title { margin: 0.4rem 0 0.5rem; font-size: 1.8rem; font-weight: 800; color: #fff; letter-spacing: -0.03em; }
.result .copy { color: var(--muted); font-size: 0.93rem; line-height: 1.6; max-width: 60ch; }

/* ---- Chips ---- */
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.7rem; }
.chip { padding: 0.4rem 0.75rem; border-radius: 999px; border: 1px solid var(--border); background: var(--surface-2); color: var(--muted); font-size: 0.8rem; font-weight: 600; }

/* ---- Footer ---- */
.foot { margin-top: 0.3rem; padding: 1rem 1.3rem; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); color: var(--muted); font-size: 0.83rem; text-align: center; }

/* ---- Streamlit widget overrides ---- */
.stButton > button, [data-testid="stFileUploaderDropzone"] button {
  border-radius: 10px !important; border: 1px solid rgba(110,231,211,0.36) !important;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: var(--ink) !important; font-weight: 700 !important; padding: 0.5rem 1.1rem !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(110,231,211,0.22) !important; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border: 1px dashed var(--border-2); background: var(--surface-2); }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); }
[data-testid="stExpander"] details { border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; background: var(--surface-2) !important; }
[data-testid="stExpander"] summary { font-weight: 600; color: var(--text); }
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stAudio"] { border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: var(--radius-sm); border: 1px solid var(--border); }

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; background: var(--surface-2); border-radius: 12px; padding: 0.3rem; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { background: rgba(110,231,211,0.10); color: #fff; }

/* Sidebar radio -> nav look */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.3rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border: 1px solid transparent; border-radius: 10px; padding: 0.5rem 0.7rem;
  transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(148,163,184,0.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(110,231,211,0.10); border-color: rgba(110,231,211,0.30);
}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p { font-weight: 600; font-size: 0.92rem; }

/* ---- Dashboard: audio info stat grid ---- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 0.7rem; margin-top: 0.9rem; }
.stat { padding: 0.8rem 0.9rem; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); }
.stat span { display: block; color: var(--muted); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
.stat strong { display: block; margin-top: 0.32rem; color: #fff; font-size: 1.02rem; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---- Dashboard: processing timeline ---- */
.timeline { display: flex; flex-direction: column; margin-top: 0.7rem; }
.tl-step { display: flex; align-items: center; gap: 0.75rem; padding: 0.38rem 0; }
.tl-ico { width: 24px; height: 24px; border-radius: 999px; display: flex; align-items: center; justify-content: center; font-size: 0.74rem; font-weight: 800; flex: none; }
.tl-ico.done { background: rgba(52,211,153,0.14); color: var(--ok); border: 1px solid rgba(52,211,153,0.42); }
.tl-ico.active { background: rgba(86,184,240,0.16); color: var(--accent-2); border: 1px solid rgba(86,184,240,0.45); animation: pulse 1.1s ease-in-out infinite; }
.tl-ico.pending { background: var(--surface-2); color: var(--faint); border: 1px solid var(--border); }
.tl-label { font-size: 0.9rem; font-weight: 600; color: var(--text); }
.tl-label.pending { color: var(--faint); font-weight: 500; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(86,184,240,0.35); } 50% { box-shadow: 0 0 0 5px rgba(86,184,240,0); } }

.proc-msg { color: var(--accent-2); font-size: 0.86rem; font-weight: 600; margin: 0.5rem 0 0.2rem; }

/* ---- Dashboard: confidence gauge ---- */
.gauge-wrap { display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap; margin-top: 0.8rem; }
.gauge { --val: 0; --col: var(--accent); width: 148px; height: 148px; border-radius: 999px; flex: none; position: relative;
  background: conic-gradient(var(--col) calc(var(--val) * 1%), rgba(148,163,184,0.14) 0);
  display: flex; align-items: center; justify-content: center; }
.gauge::before { content: ""; position: absolute; inset: 13px; border-radius: 999px; background: var(--panel); }
.gauge .g-val { position: relative; z-index: 1; font-size: 1.75rem; font-weight: 800; color: #fff; letter-spacing: -0.03em; }
.g-band { font-size: 1.02rem; font-weight: 700; color: #fff; }
.g-band.real { color: #b9f5dd; }
.g-band.fake { color: #fecdd3; }
.g-note { color: var(--muted); font-size: 0.84rem; line-height: 1.55; margin-top: 0.3rem; max-width: 30ch; }

/* ---- Dashboard: recommendation ---- */
.rec { display: flex; gap: 0.8rem; margin-top: 1rem; padding: 0.9rem 1rem; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); }
.rec-ico { width: 26px; height: 26px; border-radius: 8px; flex: none; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; background: rgba(86,184,240,0.14); color: var(--accent-2); border: 1px solid rgba(86,184,240,0.4); }
.rec strong { display: block; color: #fff; font-size: 0.88rem; font-weight: 700; }
.rec p { color: var(--muted); font-size: 0.86rem; line-height: 1.55; margin: 0.25rem 0 0; }

.viz-note { color: var(--muted); font-size: 0.85rem; line-height: 1.6; margin-top: 0.7rem; }

@media (max-width: 900px) { .block-container { padding: 1.4rem 1rem 2rem; } }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
# Reusable UI helpers
# ===========================================================================
def esc(value):
    return html.escape(str(value))


def html_block(markup):
    """Render a single HTML string as one Streamlit DOM block."""
    st.markdown(markup, unsafe_allow_html=True)


def spacer(rem=0.4):
    """Consistent vertical rhythm between blocks."""
    html_block(f'<div style="height:{rem}rem;"></div>')


# --- Composable HTML builders (return strings so full cards render as ONE block) ---
def eyebrow_html(text):
    return f'<div class="eyebrow">{esc(text)}</div>'


def section_head_html(eyebrow, title):
    return f'{eyebrow_html(eyebrow)}<h3 class="card-h">{esc(title)}</h3>'


def card_html(inner, extra=""):
    return f'<div class="card {extra}">{inner}</div>'


def metric_html(label, value, note=None):
    note_html = f"<small>{esc(note)}</small>" if note else ""
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong>{note_html}</div>'


def pill_html(text, state=""):
    return f'<div class="pill {state}"><span class="dot"></span>{esc(text)}</div>'


def confidence_row_html(label, value, cls):
    pct = max(0.0, min(100.0, float(value) * 100))
    return (
        '<div class="conf-row">'
        f'<div class="conf-top"><span>{esc(label)}</span><span>{pct:.1f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.2f}%"></div></div>'
        '</div>'
    )


def chips_html(items):
    chips = "".join(f'<div class="chip">{esc(item)}</div>' for item in items)
    return f'<div class="chips">{chips}</div>'


# --- Thin render wrappers (single-block components) ---
def page_header(title, subtitle):
    html_block(
        f'<div class="page-header"><div class="h-title">{esc(title)}</div>'
        f'<div class="h-sub">{esc(subtitle)}</div></div>'
    )


def card(title=None, copy=None, extra=""):
    inner = ""
    if title:
        inner += f"<h3>{esc(title)}</h3>"
    if copy:
        inner += f"<p>{esc(copy)}</p>"
    html_block(card_html(inner, extra))


def metric(label, value, note=None):
    html_block(metric_html(label, value, note))


def pill(text, state=""):
    html_block(pill_html(text, state))


def confidence_row(label, value, cls):
    html_block(confidence_row_html(label, value, cls))


# ===========================================================================
# Backend  ·  DO NOT MODIFY (model, features, inference are unchanged)
# ===========================================================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('deepfake_audio_model.keras')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, le, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def extract_features_from_audio(audio_data, sr):
    """Extract 26 audio features from audio data"""
    try:
        chunk_length = 3 * sr  # 3 seconds
        features_list = []

        for start in range(0, len(audio_data), chunk_length):
            end = min(start + chunk_length, len(audio_data))
            chunk = audio_data[start:end]

            if len(chunk) < sr:
                continue

            chroma_stft = np.mean(librosa.feature.chroma_stft(y=chunk, sr=sr))
            rms = np.mean(librosa.feature.rms(y=chunk))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=chunk))

            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
            mfcc_means = np.mean(mfccs, axis=1)

            chunk_features = np.array([
                chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
                zero_crossing_rate, *mfcc_means
            ])
            features_list.append(chunk_features)

        if features_list:
            return np.mean(features_list, axis=0)
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def predict_voice(model, scaler, le, features):
    """Predict if voice is REAL or FAKE"""
    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
        result = model.predict(features_scaled, verbose=0)
        pred_idx = np.argmax(result)
        prediction = le.inverse_transform([pred_idx])[0]
        confidence = result[0][pred_idx]
        fake_prob = result[0][0]
        real_prob = result[0][1]
        return prediction, confidence, fake_prob, real_prob
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None


# ===========================================================================
# Read-only loaders for evaluation artifacts (frontend display only)
# ===========================================================================
@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def result_asset(filename):
    path = os.path.join(RESULTS_DIR, filename)
    return path if os.path.exists(path) else None


# ===========================================================================
# Dashboard visualizations (display-only; do NOT affect inference)
# ===========================================================================
_VIZ_ACCENT = "#6ee7d3"
_VIZ_TEXT = "#e8eef7"
_VIZ_MUTED = "#93a1b5"
_VIZ_GRID = "#243044"

# Guards so large uploads never freeze the browser. Display-only; inference
# always runs on the full, unchanged audio.
_VIZ_MAX_SECONDS = 30
_VIZ_MAX_POINTS = 6000
_VIZ_MAX_FRAMES = 1200


def _downsample_waveform(y):
    """Reduce a long signal to a min/max peak envelope for fast, faithful plotting."""
    n = len(y)
    if n <= _VIZ_MAX_POINTS:
        return y
    bins = _VIZ_MAX_POINTS // 2
    step = n // bins
    trimmed = y[: step * bins].reshape(bins, step)
    envelope = np.empty(bins * 2, dtype=np.float32)
    envelope[0::2] = trimmed.min(axis=1)
    envelope[1::2] = trimmed.max(axis=1)
    return envelope


def _viz_window(audio_data, sr):
    """Clip audio to a display window so spectrograms stay responsive on long files."""
    max_samples = int(_VIZ_MAX_SECONDS * sr)
    if len(audio_data) > max_samples:
        return audio_data[:max_samples]
    return audio_data


def _viz_hop(n_samples):
    return max(512, int(np.ceil(n_samples / _VIZ_MAX_FRAMES)))


def _style_axes(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=_VIZ_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_VIZ_GRID)
    ax.xaxis.label.set_color(_VIZ_MUTED)
    ax.yaxis.label.set_color(_VIZ_MUTED)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)


def _style_colorbar(cbar):
    cbar.ax.yaxis.set_tick_params(color=_VIZ_MUTED, labelsize=7)
    cbar.outline.set_edgecolor(_VIZ_GRID)
    for label in cbar.ax.get_yticklabels():
        label.set_color(_VIZ_MUTED)


def waveform_figure(audio_data, sr):
    duration = len(audio_data) / sr if sr else 0
    envelope = _downsample_waveform(audio_data)
    times = np.linspace(0, duration, num=len(envelope))
    fig, ax = plt.subplots(figsize=(9, 2.7))
    fig.patch.set_alpha(0)
    ax.plot(times, envelope, color=_VIZ_ACCENT, linewidth=0.6)
    ax.fill_between(times, envelope, color=_VIZ_ACCENT, alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)
    ax.grid(color=_VIZ_GRID, alpha=0.35, linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def mel_spectrogram_figure(audio_data, sr):
    y = _viz_window(audio_data, sr)
    hop = _viz_hop(len(y))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.patch.set_alpha(0)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel frequency")
    _style_axes(ax)
    _style_colorbar(fig.colorbar(img, ax=ax, format="%+2.0f dB"))
    fig.tight_layout()
    return fig


def mfcc_figure(audio_data, sr):
    # Visualization only. Inference uses aggregated MFCC statistics from the
    # unchanged feature-extraction pipeline; this 2-D view is for display.
    y = _viz_window(audio_data, sr)
    hop = _viz_hop(len(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.patch.set_alpha(0)
    img = librosa.display.specshow(
        mfccs, sr=sr, hop_length=hop, x_axis="time", ax=ax, cmap="viridis"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")
    _style_axes(ax)
    _style_colorbar(fig.colorbar(img, ax=ax))
    fig.tight_layout()
    return fig


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def collect_audio_info(temp_path, filename, duration, sr, samples):
    channels = 1
    fmt = None
    try:
        meta = sf.info(temp_path)
        channels = meta.channels
        fmt = meta.format
    except Exception:
        pass

    size_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    bitrate_kbps = (size_bytes * 8) / duration / 1000 if duration > 0 else 0
    channel_label = {1: "Mono", 2: "Stereo"}.get(channels, f"{channels} ch")
    ext = os.path.splitext(filename)[1].lstrip(".").upper()

    return {
        "filename": filename,
        "duration": f"{duration:.2f} s",
        "sample_rate": f"{int(sr):,} Hz",
        "channels": channel_label,
        "bitrate": f"{bitrate_kbps:.0f} kbps" if bitrate_kbps else "—",
        "file_size": _human_size(size_bytes),
        "samples": f"{int(samples):,}",
        "format": fmt or ext or "WAV",
    }


def timeline_html(steps, active_idx):
    rows = []
    for i, label in enumerate(steps):
        if i < active_idx:
            state, glyph, lab_cls = "done", "✓", ""
        elif i == active_idx:
            state, glyph, lab_cls = "active", "●", ""
        else:
            state, glyph, lab_cls = "pending", str(i + 1), " pending"
        rows.append(
            f'<div class="tl-step"><div class="tl-ico {state}">{glyph}</div>'
            f'<div class="tl-label{lab_cls}">{esc(label)}</div></div>'
        )
    return card_html(
        section_head_html("Analysis pipeline", "AI processing timeline")
        + f'<div class="timeline">{"".join(rows)}</div>'
    )


def confidence_band(confidence):
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Moderate"
    return "Low"


# ===========================================================================
# Sidebar navigation
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.5rem 0.3rem 1.1rem;">
              <div class="brand">
                <div class="brand-mark">S</div>
                <div class="brand-text">
                  <strong>Sentinel</strong>
                  <span>Voice authenticity engine</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="eyebrow" style="margin:0 0.3rem 0.5rem;">Navigate</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Scanner", "Performance", "About"],
            format_func=lambda p: {"Scanner": "Scanner", "Performance": "Model performance", "About": "About project"}[p],
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow" style="margin:0.4rem 0.3rem;">At a glance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Features", "26")
        with c2:
            st.metric("Classes", "2")

        st.markdown(
            """
            <div style="margin-top:1rem; padding:0.85rem 0.95rem; border:1px solid var(--border); border-radius:12px; background:var(--surface-2);">
              <div style="color:#fff; font-weight:700; font-size:0.86rem;">Privacy-first</div>
              <div style="color:var(--muted); font-size:0.8rem; margin-top:0.25rem; line-height:1.55;">
                Audio is processed locally. Temporary files are removed after each analysis.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top:1.1rem; padding-top:0.9rem; border-top:1px solid var(--border);
                        display:flex; justify-content:space-between; align-items:center;">
              <span style="color:var(--faint); font-size:0.76rem;">Sentinel</span>
              <span class="pill" style="padding:0.2rem 0.55rem;">v1.0</span>
            </div>
            <div style="color:var(--faint); font-size:0.72rem; margin-top:0.5rem;">
              Final Year Project · Deepfake voice detection
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


# ===========================================================================
# Scanner page
# ===========================================================================
def render_input_panel():
    uploaded_file = None
    recorded_audio = None

    with st.container(border=True):
        html_block(
            section_head_html("Step 1", "Provide an audio sample")
            + '<p class="card-copy">Upload a WAV or MP3 file, or record directly from your browser.</p>'
        )

        tab_upload, tab_record = st.tabs(["Upload file", "Record voice"])

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Audio file", type=['wav', 'mp3'],
                help="Supported formats: WAV and MP3", label_visibility="collapsed",
            )
            st.caption("Best results: clear speech, at least 5 seconds, minimal background noise.")

        with tab_record:
            st.caption("Allow microphone access, speak clearly, then stop the recording.")
            audio_bytes = st.audio_input("Record a voice sample", key="audio_recorder")
            if audio_bytes:
                recorded_audio = audio_bytes
                st.success("Recording captured and ready for analysis.")

    return uploaded_file, recorded_audio


def render_status_panel(model, le, scaler):
    ready = model is not None and le is not None and scaler is not None
    state = "ok" if ready else "bad"
    text = "Model ready" if ready else "Model unavailable"
    note = (
        "Model, label encoder, and scaler are loaded."
        if ready
        else "Verify deepfake_audio_model.keras, label_encoder.pkl, and scaler.pkl."
    )
    st.markdown(
        f"""
        <div class="card tight">
          <div class="eyebrow">Step 2</div>
          <h3 style="margin:0.3rem 0 0.6rem;">System status</h3>
          <div class="pill {state}"><span class="dot"></span>{esc(text)}</div>
          <p style="color:var(--muted); font-size:0.86rem; margin:0.8rem 0 0;">{esc(note)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audio_info(info):
    order = [
        ("Filename", info["filename"]),
        ("Duration", info["duration"]),
        ("Sample rate", info["sample_rate"]),
        ("Channels", info["channels"]),
        ("Bitrate", info["bitrate"]),
        ("File size", info["file_size"]),
        ("Waveform length", info["samples"]),
        ("Format", info["format"]),
    ]
    cells = "".join(
        f'<div class="stat"><span>{esc(label)}</span>'
        f'<strong title="{esc(value)}">{esc(value)}</strong></div>'
        for label, value in order
    )
    html_block(card_html(
        section_head_html("Audio source", "Audio information")
        + f'<div class="stat-grid">{cells}</div>'
    ))


def render_audio_visuals(audio_data, sr):
    is_long = sr and len(audio_data) / sr > _VIZ_MAX_SECONDS
    window_note = (
        f' Showing the first {_VIZ_MAX_SECONDS}s for readability; the model still analyzes the full clip.'
        if is_long else ''
    )
    with st.container(border=True):
        html_block(section_head_html("Signal analysis", "Audio visualizations"))
        tab_wave, tab_mel, tab_mfcc = st.tabs(["Waveform", "Mel spectrogram", "MFCC heatmap"])

        with tab_wave:
            fig = waveform_figure(audio_data, sr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            html_block(
                '<div class="viz-note">Amplitude envelope of the audio signal over time. '
                'Natural speech shows organic variation in loudness and pacing.</div>'
            )
        with tab_mel:
            fig = mel_spectrogram_figure(audio_data, sr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            html_block(
                '<div class="viz-note">Energy distribution across mel-scaled frequency bands. '
                'Synthetic voices often reveal unnatural harmonic structure or spectral smearing.'
                + window_note + '</div>'
            )
        with tab_mfcc:
            fig = mfcc_figure(audio_data, sr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            html_block(
                '<div class="viz-note">Mel-frequency cepstral coefficients over time — a compact view '
                'of timbre. Visualization only; the model consumes aggregated MFCC statistics from the '
                'unchanged feature pipeline.' + window_note + '</div>'
            )


def run_analysis_pipeline(audio_data, sr, model, scaler, le):
    """Animate the processing timeline while running the real, unchanged inference."""
    steps = [
        "Audio uploaded", "Audio validated", "Waveform generated", "Spectrogram generated",
        "Feature extraction", "Neural network analysis", "Confidence calculation", "Final prediction",
    ]
    messages = [
        "Reading audio stream...", "Validating signal integrity...", "Rendering waveform...",
        "Building mel spectrogram...", "Extracting acoustic features...", "Running neural network...",
        "Calculating confidence...", "Generating result...",
    ]

    timeline = st.empty()
    bar = st.progress(0)
    status = st.empty()

    features = prediction = confidence = fake_prob = real_prob = None
    for i, _ in enumerate(steps):
        timeline.markdown(timeline_html(steps, i), unsafe_allow_html=True)
        status.markdown(f'<div class="proc-msg">{esc(messages[i])}</div>', unsafe_allow_html=True)

        if i == 4:  # real feature extraction
            features = extract_features_from_audio(audio_data, sr)
        elif i == 5 and features is not None:  # real inference
            prediction, confidence, fake_prob, real_prob = predict_voice(model, scaler, le, features)

        bar.progress(int((i + 1) / len(steps) * 100))
        time.sleep(0.32)

    timeline.markdown(timeline_html(steps, len(steps)), unsafe_allow_html=True)
    status.empty()
    bar.empty()
    return features, prediction, confidence, fake_prob, real_prob


def render_prediction_dashboard(prediction, confidence, fake_prob, real_prob):
    is_real = prediction == "REAL"
    cls = "real" if is_real else "fake"
    title = "Authentic voice" if is_real else "Synthetic voice"
    risk = "Low risk" if is_real else "High risk"
    band = confidence_band(confidence)
    narrative = (
        "Acoustic pattern is consistent with natural human speech."
        if is_real
        else "Acoustic pattern is consistent with generated or converted speech."
    )

    html_block(
        f'<div class="result {cls}">'
        f'<div class="eyebrow">Prediction</div>'
        f'<div class="title">{esc(title)}</div>'
        f'<div class="copy">{esc(narrative)}</div>'
        f'<div style="display:flex; gap:0.5rem; margin-top:0.9rem; flex-wrap:wrap;">'
        f'{pill_html(risk, cls)}{pill_html(f"Confidence {confidence:.1%}")}'
        f'{pill_html(f"{band} certainty")}</div></div>'
    )

    spacer(0.6)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric("Prediction", prediction)
    with c2:
        metric("Confidence", f"{confidence:.1%}")
    with c3:
        metric("Risk level", risk)

    spacer(0.6)
    gauge_col = "var(--ok)" if is_real else "var(--bad)"
    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        html_block(card_html(
            eyebrow_html("Confidence")
            + '<div class="gauge-wrap">'
              f'<div class="gauge" style="--val:{confidence * 100:.1f}; --col:{gauge_col};">'
              f'<div class="g-val">{confidence * 100:.0f}%</div></div>'
              f'<div><div class="g-band {cls}">{esc(band)} confidence</div>'
              '<div class="g-note">Probability the model assigns to the predicted class.</div></div>'
              '</div>'
        ))
    with right:
        html_block(card_html(
            eyebrow_html("Probability breakdown")
            + confidence_row_html("Synthetic (FAKE)", fake_prob, "fake")
            + confidence_row_html("Authentic (REAL)", real_prob, "real")
        ))

    spacer(0.6)
    summary = (
        "The uploaded audio has been analyzed successfully. The extracted acoustic features indicate "
        f"{'a natural human voice' if is_real else 'a high probability of synthetic speech'}. "
        f"Model confidence is {band.lower()} at {confidence:.1%}."
    )
    if is_real:
        recommendation = (
            "No manipulation indicators were detected. For sensitive use cases, combine this result "
            "with additional verification rather than relying on it alone."
        )
    else:
        recommendation = (
            "Treat this audio as untrusted. Do not use it for voice authentication or as proof of "
            "identity, and consider flagging its source for review."
        )
    html_block(card_html(
        section_head_html("Analysis summary", "What the model found")
        + f'<p class="card-copy" style="margin-bottom:0;">{esc(summary)}</p>'
        + f'<div class="rec"><div class="rec-ico">!</div><div><strong>Recommendation</strong>'
          f'<p>{esc(recommendation)}</p></div></div>'
    ))

    spacer(0.6)
    html_block(card_html(
        eyebrow_html("Features analyzed")
        + chips_html([
            "Chroma STFT", "RMS energy", "Spectral centroid", "Spectral bandwidth",
            "Spectral rolloff", "Zero crossing rate", "MFCC 1-20", "StandardScaler normalized",
        ])
    ))


def render_model_metrics():
    metrics = load_metrics()
    if not metrics:
        return

    spacer(0.6)
    html_block(card_html(section_head_html(
        "Benchmark", "Model performance on the held-out test set"
    )))
    spacer(0.3)
    cols = st.columns(5)
    stats = [
        ("Accuracy", f"{metrics.get('accuracy', 0):.1%}"),
        ("Precision", f"{metrics.get('precision_macro', 0):.2f}"),
        ("Recall", f"{metrics.get('recall_macro', 0):.2f}"),
        ("F1 score", f"{metrics.get('f1_score_macro', 0):.2f}"),
        ("ROC AUC", f"{metrics.get('roc_auc_fake', 0):.2f}"),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            metric(label, value)


def render_empty_state():
    st.markdown(
        """
        <div class="card">
          <div class="eyebrow">No audio yet</div>
          <h3 style="margin:0.4rem 0 0.4rem;">Upload or record to begin</h3>
          <p>The prediction, confidence breakdown, and analyzed feature summary will appear here after analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("How it works"):
        st.markdown(
            """
            1. Provide a WAV or MP3 file, or record a voice sample.
            2. The app extracts 26 acoustic features from the audio.
            3. The trained model returns an authenticity decision with confidence.

            For best results, use clear speech with minimal background noise and at least 5 seconds of audio.
            """
        )


def process_audio(audio_to_process, audio_source, model, scaler, le):
    temp_path = "temp_audio.wav"
    try:
        with open(temp_path, "wb") as f:
            if audio_source == "upload":
                f.write(audio_to_process.getbuffer())
            else:
                if isinstance(audio_to_process, bytes):
                    f.write(audio_to_process)
                else:
                    f.write(audio_to_process.getvalue() if hasattr(audio_to_process, 'getvalue') else audio_to_process)

        with st.spinner("Loading audio"):
            audio_data, sr = librosa.load(temp_path, sr=None)
        duration = len(audio_data) / sr
        filename = audio_to_process.name if audio_source == "upload" else "Browser recording"

        # Section 1 — audio information
        info = collect_audio_info(temp_path, filename, duration, sr, len(audio_data))
        render_audio_info(info)

        # Sections 2-4 — waveform, mel spectrogram, MFCC heatmap (before prediction)
        spacer(0.6)
        render_audio_visuals(audio_data, sr)

        # Sections 5-6 — animated timeline + progress running the real inference
        spacer(0.6)
        features, prediction, confidence, fake_prob, real_prob = run_analysis_pipeline(
            audio_data, sr, model, scaler, le
        )

        if features is None:
            st.error("Could not extract features from this audio file.")
            return
        if prediction is None:
            return

        # Sections 7-9 — prediction dashboard, confidence gauge, summary, recommendation
        spacer(0.6)
        render_prediction_dashboard(prediction, confidence, fake_prob, real_prob)

        # Section 10 — real model metrics
        render_model_metrics()

        spacer(0.4)
        st.success("Analysis complete.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def page_scanner(model, le, scaler):
    page_header("Voice authenticity scanner", "Upload or record a voice sample to classify it as authentic or synthetic.")

    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        uploaded_file, recorded_audio = render_input_panel()
    with right:
        render_status_panel(model, le, scaler)

    audio_to_process = None
    audio_source = None
    if uploaded_file is not None:
        audio_to_process, audio_source = uploaded_file, "upload"
    elif recorded_audio is not None:
        audio_to_process, audio_source = recorded_audio, "record"

    if audio_to_process is not None and model is not None and scaler is not None and le is not None:
        process_audio(audio_to_process, audio_source, model, scaler, le)
    elif audio_to_process is not None:
        st.error("Model artifacts are not available. Verify the model, scaler, and label encoder files.")
    else:
        render_empty_state()


# ===========================================================================
# Performance page (renders real evaluation artifacts, read-only)
# ===========================================================================
def page_performance():
    page_header("Model performance", "Evaluation on the held-out test set, generated during training.")

    metrics = load_metrics()
    if not metrics:
        card("Metrics unavailable", "Run the training pipeline to generate results/evaluation_metrics.json.")
        return

    report = metrics.get("classification_report", {})
    fake = report.get("FAKE", {})
    real = report.get("REAL", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
    with c2:
        metric("Macro F1", f"{metrics.get('f1_score_macro', 0):.2f}")
    with c3:
        metric("ROC AUC", f"{metrics.get('roc_auc_fake', 0):.2f}", "FAKE class")
    with c4:
        metric("Test samples", f"{int(metrics.get('test_samples', 0)):,}", "audio chunks")

    spacer(0.5)
    html_block(card_html(section_head_html("Per-class breakdown", "Precision, recall & F1")))

    spacer(0.3)
    cA, cB = st.columns(2)
    with cA:
        html_block(card_html(
            eyebrow_html("FAKE (synthetic)")
            + confidence_row_html("Precision", fake.get("precision", 0), "fake")
            + confidence_row_html("Recall", fake.get("recall", 0), "fake")
            + confidence_row_html("F1-score", fake.get("f1-score", 0), "fake")
            + f'<p class="support-note">Support: {int(fake.get("support", 0)):,} chunks</p>'
        ))
    with cB:
        html_block(card_html(
            eyebrow_html("REAL (authentic)")
            + confidence_row_html("Precision", real.get("precision", 0), "real")
            + confidence_row_html("Recall", real.get("recall", 0), "real")
            + confidence_row_html("F1-score", real.get("f1-score", 0), "real")
            + f'<p class="support-note">Support: {int(real.get("support", 0)):,} chunks</p>'
        ))

    spacer(0.5)
    st.info(
        "The test set is heavily imbalanced (many more synthetic than authentic chunks), "
        "so REAL-class precision is limited. Accuracy alone overstates real-world performance — "
        "review per-class metrics and the confusion matrix below."
    )

    spacer(0.5)
    tab_cm, tab_curves, tab_train = st.tabs(["Confusion matrix", "ROC & PR curves", "Training history"])

    with tab_cm:
        img = result_asset("confusion_matrix.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.caption("confusion_matrix.png not found.")

    with tab_curves:
        cc1, cc2 = st.columns(2)
        with cc1:
            roc = result_asset("roc_curve.png")
            st.image(roc, use_container_width=True) if roc else st.caption("roc_curve.png not found.")
        with cc2:
            pr = result_asset("precision_recall_curve.png")
            st.image(pr, use_container_width=True) if pr else st.caption("precision_recall_curve.png not found.")

    with tab_train:
        tc1, tc2 = st.columns(2)
        with tc1:
            acc = result_asset("training_accuracy.png")
            st.image(acc, use_container_width=True) if acc else st.caption("training_accuracy.png not found.")
        with tc2:
            loss = result_asset("training_loss.png")
            st.image(loss, use_container_width=True) if loss else st.caption("training_loss.png not found.")


# ===========================================================================
# About page
# ===========================================================================
def page_about():
    page_header("About the project", "A deep-learning system for detecting synthetic (deepfake) voices from acoustic features.")

    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        card(
            "Overview",
            "Sentinel analyzes short voice samples and classifies them as authentic human speech or "
            "AI-generated / converted speech. It extracts 26 acoustic features per audio chunk and feeds "
            "them to a trained neural network for classification.",
        )
        spacer(0.5)
        html_block(card_html(
            section_head_html("Pipeline", "From waveform to decision")
            + '<div class="steps">'
              '1. Load audio and segment it into fixed-length chunks.<br>'
              '2. Extract chroma, spectral, energy and MFCC features.<br>'
              '3. Normalize features with a fitted StandardScaler.<br>'
              '4. Classify with a TensorFlow neural network.<br>'
              '5. Aggregate chunk predictions into a final decision.'
              '</div>'
        ))
    with c2:
        html_block(card_html(
            section_head_html("Tech stack", "Built with")
            + chips_html([
                "Python", "TensorFlow / Keras", "Librosa",
                "scikit-learn", "Streamlit", "NumPy",
            ])
        ))
        spacer(0.5)
        html_block(card_html(
            eyebrow_html("Disclaimer")
            + '<p style="margin-top:0.4rem;">Intended for research and academic demonstration. '
              'Predictions are bounded by the training dataset and should not be used as sole '
              'evidence of authenticity.</p>'
        ))


# ===========================================================================
# Router
# ===========================================================================
def main():
    page = render_sidebar()
    model, le, scaler = load_model()

    st.markdown('<div class="shell">', unsafe_allow_html=True)
    if page == "Scanner":
        page_scanner(model, le, scaler)
    elif page == "Performance":
        page_performance()
    else:
        page_about()

    st.markdown(
        """
        <div class="foot">
          Sentinel · Voice Authenticity Engine — Final Year Project.
          Interpret predictions within the scope of the training dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
