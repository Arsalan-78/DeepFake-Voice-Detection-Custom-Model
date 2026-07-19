import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Chunking configuration
CHUNK_DURATION_SECONDS = 5
CHUNK_OVERLAP = 0.5
SILENCE_THRESHOLD_DBFS = -50.0
AUGMENTED_AUDIO_DIR = "AUDIO_AUGMENTED"
RANDOM_SEED = 42
RESULTS_DIR = "results"


def extract_features_from_chunk(chunk, sr):
    """Extract 26 audio features from one non-silent, five-second chunk."""
    try:
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=chunk, sr=sr))
        rms = np.mean(librosa.feature.rms(y=chunk))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=chunk))

        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)

        return np.array([
            chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
            zero_crossing_rate, *mfcc_means
        ])
    except Exception as e:
        print(f"Error extracting chunk features: {e}")
        return None


def is_silent(chunk):
    """Return True when a waveform is at or below the configured dBFS floor."""
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    dbfs = float(librosa.amplitude_to_db(np.array([max(rms, 1e-12)]), ref=1.0)[0])
    return dbfs <= SILENCE_THRESHOLD_DBFS


def extract_chunks_from_audio(file_path, collect_descriptors=False):
    """Return one independent feature vector for every valid audio chunk."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []

    chunk_length = int(CHUNK_DURATION_SECONDS * sr)
    hop_length = int(chunk_length * (1 - CHUNK_OVERLAP))
    features_list = []
    descriptors = []

    for start in range(0, len(audio) - chunk_length + 1, hop_length):
        chunk = audio[start:start + chunk_length]

        if is_silent(chunk):
            continue

        features = extract_features_from_chunk(chunk, sr)
        if features is not None:
            features_list.append(features)
            if collect_descriptors:
                descriptors.append((file_path, start, sr, chunk_length))

    return features_list, descriptors


def load_audio_file_paths():
    """Collect audio file paths and labels without extracting their features."""
    paths = []
    labels = []
    extensions = ('.wav', '.mp3', '.m4a', '.flac')

    for label in ('REAL', 'FAKE'):
        directory = os.path.join("AUDIO", label)
        if not os.path.exists(directory):
            continue

        files = sorted(
            file for file in os.listdir(directory)
            if file.lower().endswith(extensions)
        )
        print(f"Found {len(files)} {label} audio files")
        for file in files:
            paths.append(os.path.join(directory, file))
            labels.append(label)

    return np.array(paths), np.array(labels)


def extract_split_features(paths, labels, split_name, collect_descriptors=False):
    """Extract independently labelled chunk features for one file-level split."""
    X = []
    y = []
    descriptors_by_label = {'REAL': [], 'FAKE': []}

    for file_path, label in tqdm(
        list(zip(paths, labels)),
        desc=f"Extracting {split_name} chunks",
        unit="file"
    ):
        chunk_features, descriptors = extract_chunks_from_audio(
            file_path,
            collect_descriptors=collect_descriptors
        )
        X.extend(chunk_features)
        y.extend([label] * len(chunk_features))
        descriptors_by_label[label].extend(descriptors)
        print(f"  {split_name}: {os.path.basename(file_path)} -> {len(chunk_features)} chunks")

    if not X:
        raise ValueError(f"No valid non-silent 5-second chunks found in the {split_name} split")

    return np.asarray(X), np.asarray(y), descriptors_by_label


def add_gaussian_noise(audio, rng):
    """Add low-level noise at a randomly selected 25–40 dB SNR."""
    signal_rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    snr_db = rng.uniform(25.0, 40.0)
    noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    return audio + rng.normal(0.0, noise_rms, size=audio.shape)


def adjust_gain(audio, rng):
    """Apply a realistic random gain adjustment."""
    gain_db = rng.uniform(-6.0, 6.0)
    return audio * (10 ** (gain_db / 20.0))


def vary_speed(audio, rng):
    """Apply a small speed change and restore the original sample count."""
    rate = rng.uniform(0.95, 1.05)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return librosa.util.fix_length(stretched, size=len(audio))


def add_light_reverb(audio, sr, rng):
    """Add several quiet, decaying early reflections."""
    reverberated = audio.astype(np.float32, copy=True)
    reflection_count = int(rng.integers(3, 7))
    delays = np.sort(rng.uniform(0.025, 0.18, size=reflection_count))
    for delay_index, delay_seconds in enumerate(delays):
        delay_samples = int(delay_seconds * sr)
        if delay_samples >= len(audio):
            continue
        decay = rng.uniform(0.05, 0.14) * np.exp(-1.2 * delay_index)
        reverberated[delay_samples:] += decay * audio[:-delay_samples]
    wet_mix = rng.uniform(0.08, 0.18)
    return (1 - wet_mix) * audio + wet_mix * reverberated


def simulate_mp3_compression(audio, sr, rng):
    """Approximate light MP3 artifacts with band-limiting and spectral quantization."""
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    nyquist = sr / 2
    cutoff = min(rng.uniform(14000.0, 18000.0), nyquist * 0.95)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    magnitude[frequencies > cutoff, :] = 0

    maximum = float(np.max(magnitude))
    if maximum > 0:
        levels = int(rng.integers(128, 257))
        magnitude = np.round(magnitude / maximum * levels) / levels * maximum

    compressed = librosa.istft(
        magnitude * np.exp(1j * phase),
        hop_length=512,
        length=len(audio)
    )
    return compressed


def augment_chunk(audio, sr, rng):
    """Randomly apply one to three realistic waveform transformations."""
    augmentations = [
        lambda x: add_gaussian_noise(x, rng),
        lambda x: adjust_gain(x, rng),
        lambda x: vary_speed(x, rng),
        lambda x: add_light_reverb(x, sr, rng),
        lambda x: simulate_mp3_compression(x, sr, rng),
    ]
    selected = rng.choice(
        len(augmentations),
        size=int(rng.integers(1, 4)),
        replace=False
    )
    augmented = audio.astype(np.float32, copy=True)
    for index in selected:
        augmented = augmentations[index](augmented)
    return np.clip(
        librosa.util.fix_length(augmented, size=len(audio)),
        -1.0,
        1.0
    ).astype(np.float32)


def generate_balancing_augmentations(descriptors_by_label, original_labels):
    """Augment only the minority training class until chunk counts are balanced."""
    counts = {
        label: int(np.sum(original_labels == label))
        for label in ('REAL', 'FAKE')
    }
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    augmentation_count = counts[majority_label] - counts[minority_label]

    if augmentation_count <= 0:
        return np.empty((0, 26)), np.empty((0,), dtype=str), 0

    sources = descriptors_by_label[minority_label]
    if not sources:
        raise ValueError(f"No {minority_label} training chunks are available for augmentation")

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.integers(0, len(sources), size=augmentation_count)
    selected_sources = [sources[index] for index in selected_indices]
    grouped_sources = {}
    for output_index, descriptor in enumerate(selected_sources):
        grouped_sources.setdefault(descriptor[0], []).append((output_index, descriptor))

    output_directory = os.path.join(AUGMENTED_AUDIO_DIR, minority_label)
    os.makedirs(output_directory, exist_ok=True)

    augmented_features = []
    augmented_labels = []
    with tqdm(total=augmentation_count, desc=f"Augmenting {minority_label}", unit="chunk") as bar:
        for file_path, entries in grouped_sources.items():
            audio, loaded_sr = librosa.load(file_path, sr=None)
            stem = os.path.splitext(os.path.basename(file_path))[0]

            for output_index, (_, start, sr, chunk_length) in entries:
                if loaded_sr != sr:
                    raise ValueError(f"Sample rate changed while reading {file_path}")
                chunk = audio[start:start + chunk_length]
                augmented = augment_chunk(chunk, sr, rng)
                output_path = os.path.join(
                    output_directory,
                    f"{stem}_aug_{output_index:06d}.wav"
                )
                sf.write(output_path, augmented, sr, subtype='PCM_16')

                features = extract_features_from_chunk(augmented, sr)
                if features is not None:
                    augmented_features.append(features)
                    augmented_labels.append(minority_label)
                bar.update(1)

    return (
        np.asarray(augmented_features),
        np.asarray(augmented_labels),
        len(augmented_features)
    )


def prepare_data(paths, labels):
    """Create leakage-safe file-level splits, then extract and scale chunks."""
    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=42,
        stratify=remaining_labels
    )

    print(
        f"\nFile split: {len(train_paths)} train, "
        f"{len(val_paths)} validation, {len(test_paths)} test"
    )

    X_train, y_train_labels, train_descriptors = extract_split_features(
        train_paths, train_labels, "train", collect_descriptors=True
    )
    X_val, y_val_labels, _ = extract_split_features(
        val_paths, val_labels, "validation"
    )
    X_test, y_test_labels, _ = extract_split_features(
        test_paths, test_labels, "test"
    )

    original_train_count = len(X_train)
    X_augmented, y_augmented, augmented_count = generate_balancing_augmentations(
        train_descriptors,
        y_train_labels
    )
    if augmented_count:
        X_train = np.concatenate([X_train, X_augmented])
        y_train_labels = np.concatenate([y_train_labels, y_augmented])

    # Fit only on training chunks to prevent validation/test leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    le.fit(train_labels)

    print("Label Encoding:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")

    y_train = to_categorical(le.transform(y_train_labels), num_classes=len(le.classes_))
    y_val = to_categorical(le.transform(y_val_labels), num_classes=len(le.classes_))
    y_test = to_categorical(le.transform(y_test_labels), num_classes=len(le.classes_))

    print(
        f"Chunk samples: {len(X_train)} train, "
        f"{len(X_val)} validation, {len(X_test)} test"
    )
    print("\nAugmentation summary")
    print(f"  Original recordings: {len(paths)}")
    print(
        f"  Original chunks created: "
        f"{original_train_count + len(X_val) + len(X_test)}"
    )
    print(f"  Augmented samples: {augmented_count}")
    print(f"  Final REAL training samples: {np.sum(y_train_labels == 'REAL')}")
    print(f"  Final FAKE training samples: {np.sum(y_train_labels == 'FAKE')}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler


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


def save_training_history(history):
    """Save training/validation accuracy and loss plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history.history['accuracy'], label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_accuracy.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history.history['loss'], label='Training loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=300)
    plt.close()

    pd.DataFrame(history.history).to_csv(
        os.path.join(RESULTS_DIR, 'training_history.csv'),
        index_label='epoch'
    )


def evaluate_model(model, X_test, y_test, label_encoder, history):
    """Evaluate once on the held-out test set and persist all results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = list(label_encoder.classes_)
    fake_index = int(label_encoder.transform(['FAKE'])[0])
    fake_true = (y_true == fake_index).astype(int)
    fake_scores = y_pred[:, fake_index]

    cm = confusion_matrix(y_true, y_pred_labels, labels=range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
    false_positive_rate, true_positive_rate, _ = roc_curve(fake_true, fake_scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    pr_precision, pr_recall, _ = precision_recall_curve(fake_true, fake_scores)
    pr_auc = auc(pr_recall, pr_precision)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report_text)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f'FAKE ROC (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        pr_recall,
        pr_precision,
        label=f'FAKE Precision-Recall (AUC = {pr_auc:.4f})'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'), dpi=300)
    plt.close()

    metrics = {
        'test_loss': float(loss),
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_score_macro': float(f1),
        'roc_auc_fake': float(roc_auc),
        'precision_recall_auc_fake': float(pr_auc),
        'test_samples': int(len(y_true)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(RESULTS_DIR, 'evaluation_metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=4)
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as file:
        file.write(report_text)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(RESULTS_DIR, 'classification_report.csv')
    )

    save_training_history(history)
    print(f"Saved evaluation metrics and plots to {RESULTS_DIR}/")


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

    paths, labels = load_audio_file_paths()
    
    if len(paths) < 10:
        print("❌ Not enough audio files to train!")
        print("Please add more audio files to AUDIO/REAL and AUDIO/FAKE folders")
        exit(1)
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = prepare_data(
        paths, labels
    )

    model = build_model(X_train.shape[1])
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop]
    )

    evaluate_model(model, X_test, y_test, label_encoder, history)

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
    print(f"✅ Model trained on {len(X_train)} five-second audio chunks")
    print("✅ Now you can test with test_audio.py")
    print("="*50)
