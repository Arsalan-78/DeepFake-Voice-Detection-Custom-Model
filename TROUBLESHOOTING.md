# Troubleshooting & FAQ

## Quick Index

- [Installation Issues](#installation-issues)
- [GPU/CUDA Issues](#gpu-cuda-issues)
- [Audio Processing Problems](#audio-processing-problems)
- [Model Training Issues](#model-training-issues)
- [Prediction Errors](#prediction-errors)
- [Streamlit Web UI Problems](#streamlit-web-ui-problems)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Performance Tips](#performance-tips)

---

## Installation Issues

### Problem: `pip install tensorflow` fails

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
```bash
# Windows: Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"

# Or use pre-built wheel
pip install --upgrade pip
pip install tensorflow --only-binary :all:
```

### Problem: `ImportError: No module named 'tensorflow'`

**Solution**:
```bash
# Verify virtual environment is activated
source tf_env/bin/activate  # macOS/Linux
tf_env\Scripts\activate  # Windows

# Install again
pip install tensorflow

# Verify
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Problem: Python version incompatibility

**Error**: `This Python (3.11) is not compatible with TensorFlow 2.16`

**Solution**:
```bash
# Check Python version
python --version

# If not 3.12, install correct version or create new env
# Using conda (recommended)
conda create -n tf_env python=3.12
conda activate tf_env
pip install -r requirements.txt
```

### Problem: `ModuleNotFoundError: No module named 'librosa'`

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Individual installation
pip install librosa

# Verify
python -c "import librosa; print(librosa.__version__)"
```

---

## GPU/CUDA Issues

### Problem: TensorFlow detects CPU only (no GPU)

**Error**: `Could not load dynamic library 'cudart64_*.dll'`

**Solution**:
```bash
# 1. Check if GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty list, GPU not detected

# 2. Install GPU-enabled TensorFlow
pip install tensorflow[and-cuda]

# 3. Verify CUDA installation
nvidia-smi  # Should show GPU info

# 4. Check NVIDIA drivers
# Update drivers from: https://www.nvidia.com/Download/driverDetails.aspx
```

**For RTX 3050 users**:
```bash
# Ensure CUDA 12.x is installed
# TensorFlow 2.16 requires CUDA 12.x

# Verify CUDA version
nvcc --version

# If CUDA not installed, download from:
# https://developer.nvidia.com/cuda-downloads
```

### Problem: Out of memory errors with GPU

**Error**: `CUDA out of memory. Tried to allocate X.XX GiB`

**Solution**:
```python
# Method 1: Reduce batch size
model.fit(X_train, y_train,
          batch_size=4,  # Reduce from 8
          epochs=100)

# Method 2: Limit GPU memory
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Method 3: Use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Problem: GPU works but training is slow

**Diagnosis**:
```bash
# Check GPU usage during training
watch -n 1 nvidia-smi

# Check TensorFlow GPU utilization in code
import tensorflow as tf
print("GPU Available:", tf.test.is_built_with_cuda())
print("GPU Device:", tf.config.list_physical_devices('GPU'))
```

**Solutions**:
1. Increase batch size (if memory allows)
2. Check for data preprocessing bottleneck
3. Ensure GPU drivers are latest
4. Profile code to find slow operations

---

## Audio Processing Problems

### Problem: Audio file not found or format error

**Error**: `FileNotFoundError: [WinError 2] The system cannot find the file`

**Solution**:
```python
import os
# Verify file exists
audio_path = "path/to/audio.wav"
print(os.path.exists(audio_path))  # Should be True

# Check file format
import librosa
try:
    audio, sr = librosa.load(audio_path)
    print(f"Loaded! SR: {sr}, Duration: {len(audio)/sr:.1f}s")
except Exception as e:
    print(f"Error loading: {e}")
    # Try alternative format
    # Convert MP3 to WAV using ffmpeg if needed
```

### Problem: Supported audio formats

**Supported**: WAV, MP3, OGG, FLAC (via librosa)

**Solution for unsupported format**:
```bash
# Convert to WAV using ffmpeg
ffmpeg -i input.m4a -acodec pcm_s16le -ar 44100 output.wav

# Or use librosa to convert
import librosa
import soundfile as sf
audio, sr = librosa.load('input.m4a', sr=None)
sf.write('output.wav', audio, sr)
```

### Problem: Sample rate mismatch

**Error**: Feature extraction produces wrong shape or values

**Solution**:
```python
import librosa

# Load with specific sample rate
audio, sr = librosa.load('audio.wav', sr=22050)
print(f"Loaded at {sr} Hz")

# Or detect and resample
audio, sr_orig = librosa.load('audio.wav', sr=None)
print(f"Original SR: {sr_orig}")

# Resample if needed
if sr_orig != 22050:
    audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=22050)
```

### Problem: Audio file too long or too short

**Error**: Features have wrong dimensions

**Solution**:
```python
# Feature extraction handles variable lengths via chunking
# But optimal is 8-30 seconds

# Check audio duration
import librosa
audio, sr = librosa.load('audio.wav')
duration = len(audio) / sr
print(f"Duration: {duration:.1f} seconds")

# If too long, trim
if duration > 30:
    audio = audio[:22050*30]  # Keep first 30 seconds

# If too short, pad
if duration < 5:
    print("Warning: Audio too short, may reduce accuracy")
```

---

## Model Training Issues

### Problem: Training doesn't improve accuracy

**Symptoms**: Validation accuracy stays at ~50%

**Causes & Solutions**:

1. **Dataset issue** (most common)
```python
# Check data distribution
from collections import Counter
print(Counter(y_train))  # Should have both REAL and FAKE

# Verify features are different between classes
import numpy as np
real_features = X_train[y_train == 0]
fake_features = X_train[y_train == 1]
print(f"REAL mean: {real_features.mean(axis=0)[:5]}")
print(f"FAKE mean: {fake_features.mean(axis=0)[:5]}")
```

2. **Feature extraction issue**
```python
# Verify features are normalized
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean: {X_scaled.mean(axis=0)}")  # Should be ~0
print(f"Std: {X_scaled.std(axis=0)}")    # Should be ~1
```

3. **Model architecture problem**
```python
# Try simpler model first
model = Sequential([
    Dense(128, input_shape=(26,), activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
```

4. **Learning rate too high**
```python
# Use lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

### Problem: Model overfitting (validation acc < training acc)

**Solution**:
```python
# Increase regularization
model = Sequential([
    Dense(256, input_shape=(26,), activation='relu'),
    BatchNormalization(),
    Dropout(0.6),  # Increase dropout
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

# Or reduce model size
model = Sequential([
    Dense(64, input_shape=(26,), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
```

### Problem: Early stopping triggers immediately

**Error**: Training stops at epoch 2-3

**Solution**:
```python
# Increase patience (wait longer for improvement)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,  # Increase from 15
    restore_best_weights=True,
    min_delta=0.001  # Minimum improvement threshold
)

# Or use different metric
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Instead of val_loss
    patience=20,
    mode='max'
)
```

---

## Prediction Errors

### Problem: Model predicts only one class

**Symptom**: All predictions are FAKE or REAL

**Solution**:
```python
# 1. Verify training was balanced
# Check training data at beginning of main.py

# 2. Scale input features with same scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
test_features = scaler.transform(test_features)

# 3. Check confidence scores
predictions = model.predict(test_features)
print(predictions)  # Should show probability distribution

# 4. Analyze model weights
model.summary()  # Check if model is learning
```

### Problem: High confidence wrong predictions

**Symptom**: Model predicts FAKE with 99% for REAL audio

**Root cause**: Feature mismatch between training and testing

**Solution**:
```python
# Verify identical feature extraction
# Check that test_audio.py uses same extract_features() as main.py

# Debug: Print features for known good and known bad
features_good = extract_features_from_audio("test_good.wav")
features_bad = extract_features_from_audio("test_bad.wav")
print(f"Good: {features_good[:5]}")
print(f"Bad: {features_bad[:5]}")

# Compare with training data
print(f"Train REAL mean: {X_train[y_train==1].mean(axis=0)[:5]}")
print(f"Train FAKE mean: {X_train[y_train==0].mean(axis=0)[:5]}")
```

### Problem: `IndexError` in prediction

**Error**: `IndexError: index 1 is out of bounds for axis 0 with size 1`

**Solution**:
```python
# Ensure proper input shape
features = extract_features_from_audio(audio_path)
print(features.shape)  # Should be (26,)

# Add batch dimension for model.predict()
features = np.expand_dims(features, axis=0)
print(features.shape)  # Should be (1, 26)

# Then predict
predictions = model.predict(features)
print(predictions.shape)  # Should be (1, 2)
```

---

## Streamlit Web UI Problems

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
# Install streamlit
pip install streamlit

# Verify
streamlit --version
```

### Problem: File upload not working

**Error**: Upload button appears but files don't process

**Solution** in `app.py`:
```python
# Ensure file handling is correct
uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.getbuffer())
        audio_path = tmp.name
    
    # Process
    try:
        prediction = predict_voice(audio_path)
    except Exception as e:
        st.error(f"Error: {str(e)}")
```

### Problem: Streamlit app runs but shows blank page

**Solution**:
```bash
# 1. Run with verbose output
streamlit run app.py --logger.level=debug

# 2. Check for syntax errors
python -m py_compile app.py

# 3. Run simple test
streamlit hello  # Should work
```

### Problem: App is slow/times out

**Error**: "App is not responding" or loading takes >30s per prediction

**Solution**:
```python
# Cache model to avoid reloading
import streamlit as st
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('deepfake_audio_model.keras')

model = load_model()

# Use asyncio for long tasks
import asyncio
async def predict_async(audio_path):
    return predict_voice(audio_path)

# Show progress
with st.spinner("Analyzing..."):
    result = predict_voice(audio_path)
```

### Problem: CSS styling not appearing

**Solution**:
```python
# Ensure CSS is in st.markdown() with unsafe_allow_html=True
st.markdown("""
<style>
    body {
        background: #0d1117;
    }
</style>
""", unsafe_allow_html=True)
```

---

## Frequently Asked Questions

### Q: What's the accuracy of the model?

**A**: **86.67%** on test set (15 samples)
- FAKE detection: 100% (11/11 correct)
- REAL detection: 50% (2/4 correct)
- Recommend collecting more REAL samples to improve

### Q: Why do you have more FAKE samples than REAL?

**A**: Real-world deepfake datasets often have more synthetic samples. This reflects deployment scenarios. The imbalance is handled via stratified splitting and balanced batch formation.

### Q: Can this detect ALL types of deepfakes?

**A**: No. This model is trained on:
- Voice cloning (Coqui TTS)
- Text-to-speech (MSTTS, Google TTS)
- Voice conversion (RealTime VC)

Other synthesis methods may not be detected accurately.

### Q: Why not use a pre-trained model like Wav2Vec?

**A**: Pre-trained models require:
- Large GPU memory (often 12GB+)
- Longer inference time
- Less interpretable results

Custom features + small model = fast, efficient, transparent.

### Q: Is the model privacy-preserving?

**A**: Yes:
- ✅ Runs locally (no cloud upload)
- ✅ No internet required
- ✅ Audio not stored
- ✅ Open-source code

### Q: Can this work for languages other than English?

**A**: Partially. Model trained mostly on English (82.7% of samples). May work for:
- Similar languages (German, Scandinavian)
- Phonetically similar accents

Not tested for: Arabic, Chinese, Hindi, etc.

### Q: How long does prediction take?

**A**: 
- CPU: 2-5 seconds
- GPU (RTX 3050): 1-2 seconds
- Cloud (CPU): 5-8 seconds + overhead

### Q: Can I train my own model?

**A**: Yes:
1. Collect audio samples in AUDIO/REAL and AUDIO/FAKE
2. Run `python main.py`
3. Model automatically trains and saves

### Q: How do I improve the accuracy?

**A**: 
1. **Collect more REAL samples** (target: 100+)
2. **Add diverse FAKE types** (different TTS engines)
3. **Try new features** (prosody, mel-spectrogram)
4. **Expand to other languages**

### Q: What if I need a REST API?

**A**: See [DEPLOYMENT.md](./DEPLOYMENT.md) for:
- FastAPI endpoint
- AWS Lambda deployment
- GCP Cloud Run setup

### Q: Can this run on my phone?

**A**: 
- Model is 5MB (lite version possible)
- TensorFlow Lite can convert to mobile
- See [DEPLOYMENT.md](./DEPLOYMENT.md) for mobile setup

### Q: How do I cite this project?

**A**:
```bibtex
@software{fake_voice_detection_2026,
  title={Voice Deepfake Detection System},
  author={Arsalan and Contributors},
  year={2026},
  url={https://github.com/yourusername/fake-voice-detection}
}
```

---

## Performance Tips

### 1. Speed up training
```python
# Use mixed precision (speeds up 2-3x on GPU)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Reduce validation frequency
model.fit(..., validation_freq=5)  # Validate every 5 epochs
```

### 2. Reduce inference latency
```python
# Quantize model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Load TFLite instead of Keras
interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()
```

### 3. Batch process audio
```python
# Process 10 files at once
def batch_predict(audio_files):
    features = [extract_features_from_audio(f) for f in audio_files]
    features = np.array(features)
    predictions = model.predict(features, batch_size=8)
    return predictions
```

### 4. Memory optimization
```python
# Clear unused tensors
import gc
gc.collect()
tf.keras.backend.clear_session()

# Limit dataset caching
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
```

---

## Still Having Issues?

### Debug Checklist
- [ ] Python 3.12+ installed?
- [ ] Virtual environment activated?
- [ ] All requirements installed? (`pip list`)
- [ ] Audio file exists and is accessible?
- [ ] Model file exists? (`ls deepfake_audio_model.keras`)
- [ ] GPU detected? (`nvidia-smi`)
- [ ] Enough disk space? (500MB+)
- [ ] Enough RAM? (8GB+)

### Report Issues
If still stuck:
1. Check existing [GitHub Issues](../../issues)
2. Create new issue with:
   - Error message (full traceback)
   - Python version (`python --version`)
   - TensorFlow version (`python -c "import tensorflow; print(tensorflow.__version__)"`)
   - OS (Windows/Mac/Linux)
   - Steps to reproduce
3. Mention if it's GPU or CPU related

### Community Help
- [TensorFlow Issues](https://github.com/tensorflow/tensorflow/issues)
- [Librosa Docs](https://librosa.org/)
- [Streamlit Forum](https://discuss.streamlit.io/)

---

**Last Updated**: April 8, 2026  
**Maintained By**: Arsalan & Contributors
