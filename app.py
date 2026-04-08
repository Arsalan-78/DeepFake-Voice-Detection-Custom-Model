import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import librosa
import os
import time
import tempfile
import base64
from pathlib import Path
from io import BytesIO

st.set_page_config(
    page_title="🎤 Voice Deepfake Detector",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f1f2e 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stSidebarNav"] {
        background: transparent;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 62, 0.95) 100%);
        border-right: 2px solid #00d9ff;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main {
        padding: 2rem;
        background: transparent;
    }
    
    .main-header {
        text-align: center;
        padding: 40px 30px;
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 157, 211, 0.1) 100%);
        border: 2px solid #00d9ff;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(0, 217, 255, 0.05) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        color: #00d9ff;
        font-size: 2.8rem;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        font-weight: 800;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: #a0d9ff;
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .result-real {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 100, 0.15) 100%);
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #00ff88;
        border-left: 6px solid #00ff88;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.3), inset 0 0 20px rgba(0, 255, 136, 0.05);
        color: #00ff88;
        backdrop-filter: blur(10px);
    }
    
    .result-fake {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 100, 100, 0.15) 100%);
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #ff4444;
        border-left: 6px solid #ff4444;
        box-shadow: 0 0 30px rgba(255, 68, 68, 0.3), inset 0 0 20px rgba(255, 68, 68, 0.05);
        color: #ff6b6b;
        backdrop-filter: blur(10px);
    }
    
    .result-real h3, .result-fake h3 {
        font-size: 2rem;
        margin-bottom: 15px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .result-real p, .result-fake p {
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .upload-box {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.8) 0%, rgba(40, 40, 80, 0.8) 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.3);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.7) 0%, rgba(40, 40, 80, 0.7) 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.08);
        border: 1px solid rgba(0, 217, 255, 0.2);
        margin: 10px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        border: 1px solid rgba(0, 217, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
        transform: translateY(-5px);
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.8) 0%, rgba(40, 40, 80, 0.8) 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        border: 1px solid rgba(0, 217, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
    }
    
    .metric-box h3 {
        color: #00d9ff;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    .probability-bar {
        height: 35px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin: 15px 0;
        overflow: hidden;
        border: 1px solid rgba(0, 217, 255, 0.2);
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.3);
    }
    
    .progress-fake {
        background: linear-gradient(90deg, #ff4444 0%, #ff8800 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 19px;
        box-shadow: 0 0 15px rgba(255, 68, 68, 0.5);
    }
    
    .progress-real {
        background: linear-gradient(90deg, #00ff88 0%, #00d9ff 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #000;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 19px;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 157, 211, 0.1) 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #00d9ff;
        border: 1px solid rgba(0, 217, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        margin: 15px 0;
        backdrop-filter: blur(10px);
        color: #a0d9ff;
        line-height: 1.8;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.5), transparent);
        margin: 30px 0;
    }
    
    .sidebar-title {
        color: #00d9ff;
        font-size: 1.3rem;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
    }
    
    /* Streamlit components styling */
    [data-testid="stMetricDelta"] {
        color: #00d9ff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 62, 0.5);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-bottom: 2px solid transparent;
        color: #a0d9ff;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.5);
        border-bottom: 2px solid #00d9ff;
        color: #00d9ff;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #00a3d3 100%);
        color: #000;
        border: none;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00d9ff 100%);
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
        transform: translateY(-2px);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d9ff 0%, #00ff88 100%);
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    .stExpander {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.8) 0%, rgba(40, 40, 80, 0.8) 100%);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
    }
    
    .stExpander > div > div > button {
        color: #00d9ff;
        font-weight: 600;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.2);
    }
    
    /* Success and error messages */
    [data-testid="stAlert"] {
        background: rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
    }
    
    input, textarea, select {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 217, 255, 0.3);
        color: #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
    }
    
    input::placeholder {
        color: rgba(160, 217, 255, 0.5);
    }
    
    /* Table styling */
    table {
        background: rgba(26, 26, 62, 0.5);
        color: #e0e0e0;
    }
    
    th {
        background: rgba(0, 217, 255, 0.1);
        color: #00d9ff;
        border-bottom: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    td {
        border-bottom: 1px solid rgba(0, 217, 255, 0.1);
    }
    
    /* Code block styling */
    code {
        background: rgba(0, 217, 255, 0.1);
        color: #00ff88;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 62, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d9ff, #00ff88);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ffff, #00ffaa);
    }
    </style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("""
    <div class="main-header">
        <h1>🎤 Voice Deepfake Detector</h1>
        <p>Advanced AI-Powered Voice Authentication System</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #667eea;">🔐 Voice Detector</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div class='sidebar-title'>ℹ️ About This System</div>", unsafe_allow_html=True)
    st.markdown("""
    **How it works:**
    1. 🎵 Upload audio OR record your voice
    2. 🔧 AI analyzes 26 features
    3. 🤖 Deep learning prediction
    
    **Supported Formats:**
    - 📁 Upload: MP3, WAV
    - 🎤 Record: Browser Microphone
    
    **Model Performance:**
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "86.67%")
   
    
    st.markdown("""
    **Features Used:**
    - 🎼 Chroma Features
    - 📊 Spectral Analysis
    - 🔊 MFCC (20 coefficients)
    - 🎯 RMS Energy
    
    **Model:** TensorFlow Neural Network
    """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #667eea;'>Made with ❤️ for Voice Security</p>", unsafe_allow_html=True)

# Load model and preprocessing tools
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
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

def extract_features_from_audio(audio_data, sr):
    """Extract 26 audio features from audio data"""
    try:
        # Segment the audio into chunks and average features
        chunk_length = 3 * sr  # 3 seconds
        features_list = []

        for start in range(0, len(audio_data), chunk_length):
            end = min(start + chunk_length, len(audio_data))
            chunk = audio_data[start:end]

            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue

            # Extract features for this chunk
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
        fake_prob = result[0][0]  # Probability of FAKE
        real_prob = result[0][1]  # Probability of REAL
        return prediction, confidence, fake_prob, real_prob
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    
    # Tabs for Upload vs Record
    tab1, tab2 = st.tabs(["📂 Upload Audio", "🎤 Record Voice"])
    
    uploaded_file = None
    recorded_audio = None
    
    with tab1:
        st.markdown("<h3 style='color: #667eea;'>Upload an audio file</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an audio file to analyze",
            type=['wav', 'mp3'],
            help="Supported: WAV, MP3 only"
        )
    
    with tab2:
        st.markdown("<h3 style='color: #667eea;'>Record your voice</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p><strong>💬 How to Record:</strong></p>
            <ol>
                <li>Click the 🎙️ microphone button below</li>
                <li>Allow browser to access your microphone</li>
                <li>Speak clearly for 5-30 seconds</li>
                <li>Click stop to finish recording</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio recorder using streamlit-webrtc alternative
        audio_bytes = st.audio_input("🎙️ Click to record", key="audio_recorder")
        
        if audio_bytes:
            recorded_audio = audio_bytes
            st.success("✅ Recording captured! Processing your audio...", icon="✅")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #667eea; display: flex; align-items: center;'><span style='font-size: 1.5rem; margin-right: 10px;'>⚙️</span> System Status</h3>", unsafe_allow_html=True)
    model, le, scaler = load_model()
    if model and le and scaler:
        st.success("✅ Model Ready", icon="✅")
        st.markdown("<p style='text-align: center; color: #27ae60; font-weight: bold;'>All systems operational</p>", unsafe_allow_html=True)
    else:
        st.error("❌ Model Error")
        st.markdown("<p style='text-align: center; color: #e74c3c;'>Please restart application</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Process uploaded file or recorded audio
audio_to_process = None
audio_source = None

if uploaded_file is not None:
    audio_to_process = uploaded_file
    audio_source = "upload"
elif recorded_audio is not None:
    audio_to_process = recorded_audio
    audio_source = "record"

if audio_to_process is not None:
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🔍 Analyzing Audio...</h2>", unsafe_allow_html=True)
    
    # Create a temporary file
    temp_path = "temp_audio.wav"
    
    try:
        # Write audio to temporary file
        with open(temp_path, "wb") as f:
            if audio_source == "upload":
                # UploadedFile object - use getbuffer()
                f.write(audio_to_process.getbuffer())
            else:  # recorded audio
                # Bytes object - write directly
                if isinstance(audio_to_process, bytes):
                    f.write(audio_to_process)
                else:
                    # Fallback for other types
                    f.write(audio_to_process.getvalue() if hasattr(audio_to_process, 'getvalue') else audio_to_process)
        
        # Load audio
        progress_bar = st.progress(0)
        with st.spinner("🎵 Loading audio file..."):
            audio_data, sr = librosa.load(temp_path, sr=None)
            audio_duration = len(audio_data) / sr
            progress_bar.progress(33)
        
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Duration</h3>
                <div class='value'>{audio_duration:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Sample Rate</h3>
                <div class='value'>{sr} Hz</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            source_name = audio_to_process.name if audio_source == "upload" else "🎤 Recording"
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Source</h3>
                <div class='value'>{source_name[:20]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Extract features
        with st.spinner("🔧 Extracting audio features..."):
            features = extract_features_from_audio(audio_data, sr)
            progress_bar.progress(66)
        
        if features is not None:
            # Make prediction
            with st.spinner("🤖 Running AI prediction..."):
                prediction, confidence, fake_prob, real_prob = predict_voice(model, scaler, le, features)
                progress_bar.progress(100)
            
            time.sleep(0.5)
            progress_bar.empty()
            
            if prediction is not None:
                # Main result display
                if prediction == "REAL":
                    st.markdown("""
                    <div class="result-real">
                        <h3>✅ REAL VOICE DETECTED</h3>
                        <p>This appears to be a genuine human voice. The AI analyzed all audio characteristics and determined this is authentic speech.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-fake">
                        <h3>⚠️ SYNTHETIC/FAKE VOICE DETECTED</h3>
                        <p>This appears to be a synthetic, AI-generated, or deepfake voice. The AI detected inconsistencies typical of generated audio.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                # Detailed metrics
                st.markdown("<h3 style='text-align: center; color: #667eea;'>📊 Detailed Analysis</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h3>Prediction</h3>
                        <div class='value' style='color: {"#27ae60" if prediction == "REAL" else "#e74c3c"};'>{prediction}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h3>Confidence</h3>
                        <div class='value' style='color: #667eea;'>{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h3>Voice Type</h3>
                        <div class='value'>{"🎤 Human" if prediction == "REAL" else "🤖 Synth"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("<h3 style='text-align: center; color: #667eea;'>📈 Probability Distribution</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style='margin: 20px 0;'>
                        <h4 style='color: #e74c3c; margin-bottom: 10px;'>🔊 FAKE Probability</h4>
                        <div class='probability-bar'>
                            <div class='progress-fake' style='width: {fake_prob*100}%;'>{fake_prob:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='margin: 20px 0;'>
                        <h4 style='color: #27ae60; margin-bottom: 10px;'>🎤 REAL Probability</h4>
                        <div class='probability-bar'>
                            <div class='progress-real' style='width: {real_prob*100}%;'>{real_prob:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                # Feature insights
                st.markdown("<h3 style='text-align: center; color: #667eea;'>🔧 Features Analyzed</h3>", unsafe_allow_html=True)
                st.markdown("""
                <div class='info-box'>
                    The AI analyzed <strong>26 audio features</strong> including:
                    <ul>
                        <li>🎼 Chroma STFT features</li>
                        <li>📊 Spectral characteristics (centroid, bandwidth, rolloff)</li>
                        <li>🔊 RMS Energy</li>
                        <li>⏸️ Zero Crossing Rate</li>
                        <li>🎯 MFCC Coefficients (20 values)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Analysis Complete!", icon="✅")
        else:
            st.error("❌ Could not extract features from audio file")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    # Show instructions when no file uploaded
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box' style='text-align: center; padding: 40px 20px;'>
        <h2 style='color: #667eea; margin-bottom: 20px;'>👆 Get Started</h2>
        <p style='font-size: 1.1rem; color: #555;'>Upload an audio file to analyze if it's REAL or FAKE voice</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Create columns for feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; text-align: center;'>🚀 Fast</h3>
            <p style='text-align: center;'>Real-time analysis with instant results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; text-align: center;'>🎯 Accurate</h3>
            <p style='text-align: center;'>86.67% accuracy on test data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; text-align: center;'>🔒 Private</h3>
            <p style='text-align: center;'>Your data is never stored or shared</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide:
        
        **Step 1: Upload**
        - Click the "Upload Audio File" button
        - Select a voice recording from your computer
        
        **Step 2: Wait**
        - The AI will analyze the audio features
        - This usually takes 2-5 seconds
        
        **Step 3: Results**
        - See if the voice is REAL or FAKE
        - View confidence score and probability
        
        ### Supported Formats:
        | Format | Extension | Quality |
        |--------|-----------|---------|
        | WAV | .wav | ⭐⭐⭐⭐⭐ Recommended |
        | MP3 | .mp3 | ⭐⭐⭐⭐ Good |
        
        ### Recording:
        - 🎙️ Use the **Record Voice** tab to capture audio directly
        - ✅ Browser microphone access required
        - 📊 Recommended: 5-30 seconds of clear voice
        
        ### Tips for Best Results:
        - ✅ Use clear, clean audio
        - ✅ Minimum 3-5 seconds of voice
        - ✅ Reduce background noise
        - ✅ Avoid very compressed audio
        - ❌ Don't use extremely low quality recordings
        """)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 30px 0; background: rgba(102, 126, 234, 0.05); border-radius: 10px;'>
    <p style='color: #667eea; font-weight: bold; font-size: 1.05rem;'>🔐 Your Privacy is Protected</p>
    <p style='color: #888;'>Audio files are processed locally and never stored on any server</p>
</div>
""", unsafe_allow_html=True)
