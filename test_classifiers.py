import os
from collections import Counter

import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


segment_length=5000
overlap=4000


# Function to extract features from audio segment
def extract_features(segment, sr):
    # MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=segment, sr=sr)
    mfcc_mean = np.mean(mfccs.T, axis=0)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_bands=5)[0]
    spectral_contrast_mean = np.mean(spectral_contrast)

    # Zero-crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    # Root Mean Square (RMS)
    rms = librosa.feature.rms (y=segment)[0]
    rms_mean = np.mean (rms)
    spectral_rolloff = librosa.feature.spectral_rolloff (y=segment, sr=sr)[0]
    spectral_rolloff_mean = np.mean (spectral_rolloff)

    return np.concatenate([mfcc_mean, [ spectral_rolloff_mean,spectral_centroid_mean]])


# Function to segment audio
def segment_audio(audio_file, segment_length=100, overlap=0.5):
    audio, sr = librosa.load (audio_file, sr=None)
    num_samples = len(audio)
    segments = []
    start = 0
    while start + segment_length <= num_samples:
        segments.append(audio[int(start):int(start + segment_length)])  # Ensure indices are integers
        start += segment_length - overlap
    return segments,sr


# Function to segment all audio files in a directory
def segment_all_audios(audio_dir, segment_length=1, overlap=0.5):
    all_segments = []
    all_labels = []
    all_srs = []

    for file_name in os.listdir (audio_dir):
        if file_name.endswith (".wav"):  # Assuming all audio files are in .wav format
            file_path = os.path.join (audio_dir, file_name)
            segments, sr = segment_audio (file_path, segment_length, overlap)
            all_segments.extend (segments)
            # Assigning labels based on file name
            # Extracting label from file name (assuming last character is the label)
            label = int (os.path.splitext (file_name)[0][-1])
            all_labels.extend ([label] * len (segments))
            all_srs.extend ([sr] * len (segments))

    return all_segments, all_labels, all_srs


# Function to train SVM classifier
def train_svm(X_train, y_train):
    svm = SVC (C=1.0,kernel='linear')
    svm.fit (X_train, y_train)
    return svm


# Function to predict labels for new audio files
def predict_new_audios(audio_dir, svm_model, segment_length=2048, overlap=1024):
    scaler = StandardScaler ()
    audio_predictions = {}

    for file_name in os.listdir (audio_dir):
        if file_name.endswith (".wav"):  # Assuming all audio files are in .wav format
            file_path = os.path.join (audio_dir, file_name)
            segments, sr = segment_audio (file_path, segment_length, overlap)

            X_new = np.array ([extract_features (segment, sr) for segment in segments])
            X_new_scaled = scaler.fit_transform (X_new)
            predictions = svm_model.predict (X_new_scaled)
            # Count occurrences of each predicted label
            label_counts = Counter (predictions)
            # Determine the label with the highest count
            predicted_label = max (label_counts, key=label_counts.get)
            audio_predictions[file_name] = predicted_label

    return audio_predictions

scaler = StandardScaler ()

# Directory containing audio files for training
train_audio_directory = "./waves/original"

# Segment all audio files for training
all_segments, all_labels, all_srs = segment_all_audios (train_audio_directory, segment_length=segment_length, overlap=overlap)

# Convert segments to feature vectors
X = np.array ([extract_features (segment, sr) for segment, sr in zip (all_segments, all_srs)])
y = np.array (all_labels)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform (X_train)
X_test_scaled = scaler.fit_transform (X_test)

# Train SVM classifier
svm_model = train_svm (X_train_scaled, y_train)

# Evaluate SVM classifier
y_pred = svm_model.predict (X_test_scaled)
accuracy = accuracy_score (y_test, y_pred)
print ("Accuracy:", accuracy)

# Directory containing new audio files for prediction
new_audio_directory = "./waves/normalized"

# Predict labels for new audio files
predictions = predict_new_audios (new_audio_directory, svm_model, segment_length=segment_length, overlap=overlap)

# Print predictions for each new audio file
for file_name, prediction in predictions.items ():
    print (f"Audio file: {file_name}, Predicted Label: {prediction}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Function to train Random Forest classifier
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier (n_estimators=100, max_depth=10, random_state=42)  # Example hyperparameters
    rf.fit (X_train, y_train)
    return rf


# Function to train Gradient Boosting classifier
def train_gradient_boosting(X_train, y_train):
    gb = GradientBoostingClassifier (n_estimators=100, learning_rate=0.1, max_depth=3,
                                     random_state=42)  # Example hyperparameters
    gb.fit (X_train, y_train)
    return gb


# Train Random Forest classifier
rf_model = train_random_forest (X_train_scaled, y_train)

# Evaluate Random Forest classifier
rf_accuracy = rf_model.score (X_test_scaled, y_test)
print ("Random Forest Accuracy:", rf_accuracy)

# Train Gradient Boosting classifier
gb_model = train_gradient_boosting (X_train_scaled, y_train)

# Evaluate Gradient Boosting classifier
gb_accuracy = gb_model.score (X_test_scaled, y_test)
print ("Gradient Boosting Accuracy:", gb_accuracy)

# Directory containing new audio files for prediction
new_audio_directory = "./waves/normalized"

# Predict labels for new audio files using the best performing classifier
rf_predictions = predict_new_audios (new_audio_directory, rf_model, segment_length=segment_length, overlap=overlap)
gb_predictions = predict_new_audios (new_audio_directory, gb_model, segment_length=segment_length, overlap=overlap)

# Print predictions for each new audio file
for file_name, prediction in rf_predictions.items ():
    print (f"Random Forest Prediction for {file_name}: {prediction}")

for file_name, prediction in gb_predictions.items ():
    print (f"Gradient Boosting Prediction for {file_name}: {prediction}")


from sklearn.neighbors import KNeighborsClassifier

# Function to train K Nearest Neighbors classifier
def train_knn(X_train, y_train, n_neighbors=6):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Train KNN classifier
knn_model = train_knn(X_train_scaled, y_train)

# Evaluate KNN classifier
knn_accuracy = knn_model.score(X_test_scaled, y_test)
print("KNN Accuracy:", knn_accuracy)

# Predict labels for new audio files using KNN classifier
knn_predictions = predict_new_audios(new_audio_directory, knn_model, segment_length=segment_length, overlap=overlap)

# Print predictions for each new audio file
for file_name, prediction in knn_predictions.items():
    print(f"KNN Prediction for {file_name}: {prediction}")
