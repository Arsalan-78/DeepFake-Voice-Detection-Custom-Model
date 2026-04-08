# Contributing to Voice Deepfake Detection

Thank you for your interest in contributing to this research project! This document provides guidelines for how to contribute effectively.

## 🎯 How to Contribute

### 1. **Report Issues**
Found a bug or have a suggestion? 
- Open an [Issue](../../issues) with:
  - Clear description of the problem
  - Steps to reproduce (if applicable)
  - Error messages or logs
  - Suggested solution (if any)

### 2. **Add Audio Samples**
Help improve model accuracy by contributing voice samples:

#### REAL Voice Samples
- **Requirements**: 
  - Clear, natural speech (8-30 seconds)
  - 16kHz+ sample rate
  - Minimal background noise
  - Various languages/accents encouraged
- **Format**: WAV or MP3
- **License**: Must own rights or have permission
- **Submission**: Create PR adding to `AUDIO/REAL/` folder

#### FAKE Voice Samples (Deepfakes)
- **Source Types**:
  - Voice cloning results
  - TTS synthesized speech
  - Voice conversion outputs
  - Audio deepfakes
- **Diversity**: Different models, languages, qualities
- **Attribution**: Document the generation method

**Submission Process**:
```bash
# Fork the repo
git clone https://github.com/yourusername/fake-voice-detection.git
cd fake-voice-detection

# Create feature branch
git checkout -b feature/add-voice-samples

# Add your audio samples
cp your_sample.wav AUDIO/REAL/  # or AUDIO/FAKE/

# Document source
# Edit AUDIO/SAMPLES.md with metadata

# Commit and push
git add .
git commit -m "Add [N] REAL/FAKE voice samples from [source]"
git push origin feature/add-voice-samples

# Create Pull Request
```

### 3. **Improve Code**
Want to enhance the implementation?

#### Code Contributions
1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Code changes** with clear commits
4. **Test thoroughly** - include test cases
5. **Follow style guide** (see below)
6. **Create PR** with description

#### Good Contribution Areas
- [ ] Data augmentation techniques
- [ ] Additional audio features
- [ ] Model architecture improvements
- [ ] Training optimization
- [ ] Real-time streaming support
- [ ] Mobile app development
- [ ] API endpoint implementation
- [ ] Documentation improvements
- [ ] Bug fixes

### 4. **Improve Documentation**
- Fix typos or clarify explanations
- Add examples or tutorials
- Improve docstrings
- Create troubleshooting guides

---

## 📋 Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/fake-voice-detection.git
cd fake-voice-detection
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv tf_env
source tf_env/bin/activate  # On Windows: tf_env\Scripts\activate

# Or using conda
conda create -n deepfake python=3.12
conda activate deepfake
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Or for GPU support (NVIDIA RTX 3050+)
pip install tensorflow[and-cuda]
```

### 4. Set Up Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

---

## 💻 Code Style Guide

### Python Style (PEP 8)
```python
# Good
def extract_features(audio_file):
    """Extract 26 audio features from file.
    
    Args:
        audio_file (str): Path to audio file
        
    Returns:
        np.array: Feature vector of shape (26,)
    """
    # Implementation
    pass

# Bad
def extractfeatures(f):
    # unclear purpose
    pass
```

### Formatting
- **Line length**: Max 100 characters
- **Indentation**: 4 spaces
- **Docstrings**: Google/NumPy style
- **Comments**: Explain WHY, not WHAT
- **Variable names**: Clear, descriptive

### Type Hints (Encouraged)
```python
from typing import Tuple, Optional
import numpy as np

def load_audio(filepath: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file and return waveform and sample rate."""
    # Implementation
    pass
```

### Testing
```python
# test_audio_processing.py
import unittest
from main import extract_features_from_audio

class TestAudioProcessing(unittest.TestCase):
    def test_feature_shape(self):
        """Test that features have correct shape."""
        # Test implementation
        pass
    
    def test_feature_range(self):
        """Test that features are within expected range."""
        # Test implementation
        pass
```

---

## 🔬 Research Contribution Guidelines

### For Paper/Algorithm Improvements

1. **Propose changes** in Issue first
   - Describe motivation
   - Explain methodology
   - Provide literature references
   - Show expected improvements

2. **Test thoroughly**
   - Run on full dataset
   - Compare with baseline
   - Include metrics (accuracy, F1, precision, recall)
   - Test on real-world audio

3. **Document changes**
   - Update RESEARCH.md
   - Add explanation comments
   - Include performance benchmarks
   - Cite references

4. **Submit PR with**
   - Code changes
   - Updated RESEARCH.md
   - Benchmark results
   - Comparison plots

### Example: Adding New Feature

```python
def extract_temporal_features(audio, sr):
    """Add temporal analysis features.
    
    Research Reference:
        - "Temporal Cues in Speech" (Author, Year)
        - Improves real voice detection by ~5%
    
    Performance:
        - Baseline F1: 0.85
        - With feature F1: 0.91
    """
    # Implementation
    pass
```

---

## 📊 Commit Message Guidelines

```
# Use imperative mood
Good:    Add voice sample augmentation
Bad:     Added voice sample augmentation
         Fixes for audio processing

# Reference issues
Good:    Fix model predictions (#42)
Bad:     Fix stuff

# Detailed message template
Type: brief description

Longer explanation if needed.
- List changes if multiple items
- Reference issues: Closes #123

Type can be:
- feat:     New feature
- fix:      Bug fix
- docs:     Documentation
- refactor: Code reorganization
- perf:     Performance improvement
- test:     Test additions
```

---

## 🧪 Testing Requirements

Before submitting PR:

```bash
# 1. Test audio loading
python test_audio.py

# 2. Run model on test set
python main.py --test

# 3. Web UI testing
streamlit run app.py

# 4. Check code quality
pip install pylint
pylint main.py test_audio.py app.py
```

---

## 📈 Data Collection Guidelines

### Contributing Voice Samples

**IMPORTANT**: Ensure you have rights to share audio

```yaml
# Add metadata to AUDIO/SAMPLES.md
- filename: sample_name.wav
  type: REAL|FAKE
  duration: "12.3s"
  sr: 44100
  language: English
  source: |
    Public speech/TTS system/Personal recording
  license: CC0|CC-BY|Custom
  notes: Optional notes about sample
```

### Audio Specifications
- Format: WAV or MP3 (MP3 acceptable)
- Sample Rate: 16kHz minimum (44.1kHz preferred)
- Duration: 5-60 seconds
- Quality: Mono or stereo, 16-bit minimum
- Noise: Minimal background noise

---

## ✅ Pull Request Process

1. **Fork** the repo
2. **Create feature branch** 
3. **Make changes** with clear commits
4. **Add tests** (if applicable)
5. **Update documentation**
6. **Push** to fork
7. **Create PR** with:
   - Clear title
   - Description of changes
   - Related issues
   - Test results
   - Benchmark improvements (if applicable)

### PR Template
```markdown
## Description
Brief explanation of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Data contribution

## Related Issues
Closes #123

## Testing
- [ ] Model trained and tested
- [ ] Tested on 5+ audio samples
- [ ] No regression in accuracy

## Changes Made
- Bulleted list of changes
- With explanations

## Benchmarks
Before: 86.67% accuracy
After: 88.90% accuracy
Improvement: +2.23%
```

---

## 🚦 Code Review Process

All PRs require:
- ✅ Code review by maintainer
- ✅ Tests passing
- ✅ Documentation updated
- ✅ No breaking changes

Reviewers will:
- Check code quality
- Verify testing
- Validate improvements
- Suggest optimizations

---

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Librosa Audio Processing](https://librosa.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Audio Deepfake Detection Research](./RESEARCH.md)
- [Project Architecture](./README.md#project-structure)

---

## 🎓 Learning Resources

### Audio Processing
- Librosa tutorials
- Audio signal processing basics
- MFCC feature engineering
- Spectrogram analysis

### Deep Learning
- TensorFlow/Keras tutorials
- Neural network architecture design
- Regularization techniques (dropout, batch norm)
- Training optimization

### Research
- ASVspoof Challenge papers
- Voice conversion and synthesis
- Speaker verification systems
- Deepfake detection methods

---

## 🤝 Community

- **Questions?** Open a [Discussion](../../discussions)
- **Found issue?** Create an [Issue](../../issues)
- **Want to chat?** [Discord/Slack invite]

---

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Acknowledgments

Thank you for contributing! We'll recognize all contributors in:
- README.md contributors section
- Release notes
- Future publications

---

**Last Updated**: April 8, 2026  
**Maintainer**: Arsalan & Contributors
