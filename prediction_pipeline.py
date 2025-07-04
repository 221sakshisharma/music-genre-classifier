import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
# Replace with your actual label order
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30, sr=22050)
    features = {}

    features['length'] = len(y)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)

    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_var'] = np.var(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_var'] = np.var(spec_bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    harm = librosa.effects.harmonic(y)
    features['harmony_mean'] = np.mean(harm)
    features['harmony_var'] = np.var(harm)

    perceptr = librosa.effects.percussive(y)
    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i - 1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i - 1])

    return pd.DataFrame([features])


def predict_genre(model, scaler, file_path):
    features = extract_features(file_path)

    # Ensure same columns as training data (e.g., if you dropped some features)
    features = features[scaler.feature_names_in_]

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_index = prediction.argmax(axis=1)[0]
    predicted_genre = genre_labels[predicted_index]
    return predicted_genre

