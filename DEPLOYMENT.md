# Deployment Guide

This guide covers deploying the Voice Deepfake Detection system to various platforms.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Streamlit Cloud](#streamlit-cloud)
3. [Docker Deployment](#docker-deployment)
4. [AWS Lambda](#aws-lambda)
5. [GCP Cloud Run](#gcp-cloud-run)
6. [Hugging Face Spaces](#hugging-face-spaces)
7. [GitHub Pages](#github-pages)
8. [Mobile Deployment](#mobile-deployment)

---

## Local Development

### Prerequisites
- Python 3.10+
- Git
- 8GB RAM
- GPU optional (RTX 3050+)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/fake-voice-detection.git
cd fake-voice-detection
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv tf_env
source tf_env/bin/activate  # Windows: tf_env\Scripts\activate

# Or using conda
conda create -n deepfake python=3.12
conda activate deepfake
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
```

### Running Locally

#### Training Mode
```bash
python main.py
# Expected output after ~8 minutes:
# Training completed!
# Model accuracy: 86.67%
```

#### Testing Mode
```bash
python test_audio.py
# Prompts for audio file path
# Returns: REAL/FAKE prediction + confidence
```

#### Web UI (Streamlit)
```bash
streamlit run app.py
# Opens browser at http://localhost:8501
```

---

## Streamlit Cloud

### Deploy in 3 Steps

1. **Create GitHub Repository**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "New app"
   - Connect GitHub account
   - Select `fake-voice-detection` repo
   - Set main file to `app.py`

3. **Deploy**
   - Click "Deploy"
   - Wait ~2-3 minutes
   - Your app is live!

### Configuration

Create `streamlit/config.toml`:
```toml
[theme]
primaryColor = "#00d9ff"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#ffffff"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
headless = true
```

### Secrets Management

Create `.streamlit/secrets.toml`:
```toml
[api]
key = "your-api-key"
endpoint = "https://your-api.com"
```

### URL
```
https://[username]-fake-voice-detection-app-[random].streamlit.app
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  deepfake-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
    volumes:
      - ./models:/app/models
      - ./AUDIO:/app/AUDIO
    restart: unless-stopped
```

### Build & Run

```bash
# Build image
docker build -t deepfake-detection:latest .

# Run container
docker run -p 8501:8501 deepfake-detection:latest

# Or use docker-compose
docker-compose up
```

### Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag deepfake-detection:latest yourusername/deepfake-detection:latest

# Push
docker push yourusername/deepfake-detection:latest
```

---

## AWS Lambda

### Serverless Deployment

1. **Install Serverless Framework**
```bash
npm install -g serverless
serverless plugin install -n serverless-python-requirements
```

2. **Create serverless.yml**

```yaml
service: deepfake-detection

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.12
  region: us-east-1
  memorySize: 3008  # 3GB
  timeout: 900  # 15 minutes
  environment:
    MODEL_PATH: ./models

functions:
  predict:
    handler: handler.predict
    events:
      - http:
          path: predict
          method: post
    layers:
      - arn:aws:lambda:us-east-1:123456789012:layer:tensorflow-layer:1

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
    layer: true
```

3. **Create handler.py**

```python
import json
import base64
import io
from test_audio import predict_audio_file
import tempfile

def predict(event, context):
    """Lambda handler for voice predictions."""
    try:
        # Get audio from request
        body = json.loads(event['body'])
        audio_base64 = body['audio']
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Predict
        result = predict_audio_file(tmp_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

4. **Deploy**
```bash
serverless deploy
```

---

## GCP Cloud Run

### Deploy to Cloud Run

1. **Create Dockerfile** (see Docker section above)

2. **Build and Push to Container Registry**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/deepfake-detection

# Or use Docker
docker build -t gcr.io/PROJECT_ID/deepfake-detection .
docker push gcr.io/PROJECT_ID/deepfake-detection
```

3. **Deploy to Cloud Run**
```bash
gcloud run deploy deepfake-detection \
  --image gcr.io/PROJECT_ID/deepfake-detection \
  --platform managed \
  --region us-central1 \
  --memory 4G \
  --timeout 3600 \
  --max-instances 10
```

4. **Get URL**
```bash
gcloud run services describe deepfake-detection --region us-central1
```

### Environment Variables
```bash
gcloud run deploy deepfake-detection \
  --set-env-vars LOG_LEVEL=INFO,MAX_UPLOAD_SIZE=200MB
```

---

## Hugging Face Spaces

### Deploy Free Alternative

1. **Create Space on Hugging Face**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `fake-voice-detection`
   - License: MIT
   - Space SDK: Docker

2. **Add Files**
```bash
git clone https://huggingface.co/spaces/yourusername/fake-voice-detection
cd fake-voice-detection
cp -r /path/to/your/repo/* .
git add .
git commit -m "Initial deployment"
git push
```

3. **Configure**

Create `README.md`:
```markdown
# Voice Deepfake Detection

Detect artificially generated and deepfake voices.

## Features
- Real-time voice analysis
- 86.67% detection accuracy
- Privacy-preserving local processing

## Usage
Upload an audio file and get instant predictions!
```

### Automatic Updates
GitHub Actions workflow to auto-sync:

```yaml
name: Sync to HuggingFace
on: [push]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: huggingface/huggingface_hub@main
        with:
          folder_path: '.'
          repo_id: 'yourusername/fake-voice-detection'
          repo_type: 'space'
          token: ${{ secrets.HF_TOKEN }}
```

---

## GitHub Pages

### Static Documentation Site

1. **Create docs/ folder**
```bash
mkdir docs
cp README.md docs/index.md
cp RESEARCH.md docs/research.md
```

2. **Create _config.yml**
```yaml
theme: jekyll-theme-minimal
title: Voice Deepfake Detection
description: Detect AI-generated and deepfake voices

nav:
  - Home: /
  - Research: /research.html
  - GitHub: https://github.com/yourusername/fake-voice-detection
```

3. **Enable GitHub Pages**
   - Go to repo Settings
   - Pages section
   - Source: Deploy from branch
   - Branch: main
   - Folder: /docs

### URL
```
https://yourusername.github.io/fake-voice-detection
```

---

## Mobile Deployment

### React Native / Expo

```bash
expo init deepfake-detection
cd deepfake-detection
```

### App Wrapper

```typescript
import * as tf from '@tensorflow/tfjs'
import { AudioContext } from 'expo-av'

export async function predictVoice(audioPath: string) {
  // Load model
  const model = await tf.loadLayersModel('file://deepfake_model')
  
  // Load audio
  const audio = await loadAudio(audioPath)
  
  // Extract features (use web audio API)
  const features = extractFeatures(audio)
  
  // Predict
  const prediction = model.predict(features)
  return prediction
}
```

---

## Performance Optimization

### Model Optimization

1. **Quantization** (Reduce size by 75%)
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("deepfake_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("deepfake_model.tflite", "wb") as f:
    f.write(tflite_model)
```

2. **Pruning** (Remove unused weights)
```python
import tensorflow_model_optimization as tfmot

pruned = tfmot.sparsity.keras.prune_low_magnitude(model)
pruned.fit(X_train, y_train)
```

### Caching

```python
# Cache model in memory
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model('deepfake_audio_model.keras')
```

### Batch Processing

```python
# Process multiple files efficiently
def batch_predict(audio_files, batch_size=32):
    model = load_model()
    predictions = []
    
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        batch_features = [extract_features(f) for f in batch]
        batch_preds = model.predict(np.array(batch_features))
        predictions.extend(batch_preds)
    
    return predictions
```

---

## Monitoring & Logging

### CloudWatch (AWS)
```python
import logging
import watchtower

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler())

logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
```

### Sentry (Error Tracking)
```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://your-sentry-dsn@sentry.io",
    traces_sample_rate=0.1
)

try:
    prediction = model.predict(features)
except Exception as e:
    sentry_sdk.capture_exception(e)
```

---

## Security Considerations

### API Authentication
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(file: UploadFile, token = Depends(security)):
    # Validate API key
    if not validate_token(token.credentials):
        raise HTTPException(status_code=401)
    
    # Process...
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile):
    # Process...
```

### Input Validation
```python
from pydantic import BaseModel, FileSize

class AudioInput(BaseModel):
    file_size: FileSize  # Max size
    format: str  # wav, mp3
    sample_rate: int  # 16000+
```

---

## Scaling Strategies

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use database for model caching
- Separate inference workers

### Vertical Scaling
- Increase container memory
- Use GPU acceleration
- Optimize batch size

### Auto-scaling Rules
```yaml
metrics:
  - CPU: > 70% → scale up
  - Memory: > 80% → scale up
  - Queue depth: > 100 → scale up
  - Target replicas: 2-10
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size, enable model quantization |
| Slow inference | Use quantized model, reduce audio chunks |
| High latency | Cache model, use GPU, implement request queuing |
| Model not loading | Check file path, verify file integrity |
| Audio format error | Convert to WAV 16kHz, check librosa version |

### Debug Mode
```bash
# Enable verbose logging
TF_CPP_MIN_LOG_LEVEL=0 streamlit run app.py --logger.level=debug
```

---

## Cost Estimation

| Platform | Monthly Cost (Small) |
|----------|-------------------|
| Streamlit Cloud | Free |
| Hugging Face Spaces | Free |
| Docker Local | $0 |
| AWS Lambda | $0-50 (pay per use) |
| GCP Cloud Run | $0-30 (pay per use) |
| AWS EC2 | $10-100 (instance dependent) |

---

## Resources

- [Streamlit Deployment](https://docs.streamlit.io/knowledge-base/deploy)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS Lambda Python](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [GCP Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy)

---

**Last Updated**: April 8, 2026  
**Maintainer**: Arsalan & Contributors
