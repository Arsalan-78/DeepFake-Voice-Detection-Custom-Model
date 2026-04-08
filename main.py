import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Extract features from audio file
def extract_features_from_audio(file_path):
    """
    Extract 26 audio features from an audio file
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    try:
        # Segment the audio into chunks and average features
        chunk_length = 3 * sr  # 3 seconds
        features_list = []

        for start in range(0, len(audio), chunk_length):
            end = min(start + chunk_length, len(audio))
            chunk = audio[start:end]

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
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Load features from audio files in AUDIO folder
def load_features_from_audio_files():
    """
    Load features from AUDIO/REAL and AUDIO/FAKE directories
    """
    print("🎤 Loading features from audio files...")
    
    X = []
    y = []
    
    # Load REAL audio files
    real_dir = "AUDIO/REAL"
    if os.path.exists(real_dir):
        files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        print(f"Found {len(files)} REAL audio files")
        for file in files:
            file_path = os.path.join(real_dir, file)
            features = extract_features_from_audio(file_path)
            if features is not None:
                X.append(features)
                y.append('REAL')
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file}")
    
    # Load FAKE audio files
    fake_dir = "AUDIO/FAKE"
    if os.path.exists(fake_dir):
        files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        print(f"Found {len(files)} FAKE audio files")
        for file in files:
            file_path = os.path.join(fake_dir, file)
            features = extract_features_from_audio(file_path)
            if features is not None:
                X.append(features)
                y.append('FAKE')
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    
    return X, y


def prepare_data(X, y):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("Label Encoding:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, le, scaler


def build_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_labels)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred_labels))


def predict_audio(model, label_encoder, file_path):
    features = extract_features(file_path)
    features = pad_features(features)
    features = features.reshape(1, -1)
    result = model.predict(features)
    label = np.argmax(result)
    decoded = label_encoder.inverse_transform([label])[0]
    print(f"Prediction for {file_path}: {decoded}")


if __name__ == "__main__":
    # Load features directly from audio files
    real_dir = "AUDIO/REAL"
    fake_dir = "AUDIO/FAKE"
    
    if not (os.path.exists(real_dir) or os.path.exists(fake_dir)):
        print(f"❌ Audio directories not found!")
        print(f"Please create:")
        print(f"  - AUDIO/REAL/ (put real voice files here)")
        print(f"  - AUDIO/FAKE/ (put fake/deepfake voice files here)")
        exit(1)

    X, y = load_features_from_audio_files()
    
    if len(X) < 10:
        print("❌ Not enough audio files to train!")
        print("Please add more audio files to AUDIO/REAL and AUDIO/FAKE folders")
        exit(1)
    
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data(X, y)

    model = build_model(X_train.shape[1])
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop]
    )

    evaluate_model(model, X_test, y_test)

    model.save("deepfake_audio_model.keras")
    print("Saved model to deepfake_audio_model.keras")

    # Save label encoder and scaler for testing
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Saved label encoder to label_encoder.pkl")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to scaler.pkl")

    print("\n" + "="*50)
    print("🎯 MODEL TRAINING COMPLETE!")
    print("="*50)
    print(f"✅ Model trained on {len(X)} audio samples")
    print("✅ Now you can test with test_audio.py")
    print("="*50)
