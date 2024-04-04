import os
import wave
from collections import Counter

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.io import wavfile
import warnings

warnings.filterwarnings ("ignore", category=UndefinedMetricWarning)
folder_path = './waves/original'
# Directory containing audio files for training
train_audio_directory = "./waves/original"
# Directory containing new audio files for prediction
normalized_audio_directory = "./waves/normalized"


# Step 1: Read wave files
def read_wave_files_in_folder(folder_path):
    """
       Read wave files from a folder.

       Args:
       - folder_path: Path to the folder containing wave files.

       Returns:
       - List of tuples containing file name and wave data dictionary.
       """
    wave_files = []
    for file_name in os.listdir (folder_path):
        if file_name.endswith ('.wav'):
            file_path = os.path.join (folder_path, file_name)
            wave_data = read_wave_file (file_path)
            if wave_data:
                wave_files.append ((file_name, wave_data))
    return wave_files


def read_wave_file(file_path):
    """
        Read a wave file and extract relevant information.

        Args:
        - file_path: Path to the wave file.

        Returns:
        - Dictionary containing wave data.
    """
    try:
        with wave.open (file_path, 'rb') as wav_file:
            num_channels = wav_file.getnchannels ()
            sample_width = wav_file.getsampwidth ()
            frame_rate = wav_file.getframerate ()
            num_frames = wav_file.getnframes ()
            signal_data = wav_file.readframes (num_frames)
            # Convert the binary data to a numpy array
            if sample_width == 1:
                signal_data = np.frombuffer (signal_data, dtype=np.uint8)
            elif sample_width == 2:
                signal_data = np.frombuffer (signal_data, dtype=np.int16)
            elif sample_width == 4:
                signal_data = np.frombuffer (signal_data, dtype=np.int32)
            # If stereo, take only one channel (you may want to mix channels for stereo)
            if num_channels == 2:
                signal_data = signal_data[::2]

            return {
                "num_channels": num_channels,
                "sample_width": sample_width,
                "frame_rate": frame_rate,
                "num_frames": num_frames,
                "signal_data": signal_data
            }
    except Exception as e:
        print ("Error reading wave file:", e)
        return None


# Step 2: Plot spectrogram of a signal
# Plot both signal waveform and spectrogram
def plot_signal_and_spectrogram(samples, fs, title=None):
    """
        Plot the signal waveform and its spectrogram.

        Args:
        - samples: Signal data.
        - fs: Frame rate.
        - title: Title for the plot (optional).
        """

    plt.figure (figsize=(10, 6))

    plt.subplot (2, 1, 1)
    plt.plot (samples)
    plt.title ('Signal Waveform')
    plt.xlabel ('Time [sec]')
    plt.ylabel ('Amplitude')

    plt.subplot (2, 1, 2)
    plt.specgram (samples, Fs=fs)
    plt.title ('Spectrogram')
    plt.ylabel ('Frequency [Hz]')
    plt.xlabel ('Time [sec]')

    if title:
        plt.suptitle (title)

    plt.tight_layout ()
    plt.show ()


# Step 3: Generate mixing matrix
def generate_mixing_matrix():
    """
       Generate a mixing matrix.

       Returns:
       - Mixing matrix.
       """
    A = np.random.uniform (0.5, 2.5, (6, 6))
    return A


# Step 4: Create mixed signals
def create_mixed_signals(audio_sources, mixing_matrix):
    """
        Create mixed signals by applying a mixing matrix.

        Args:
        - audio_sources: Array of audio signals.
        - mixing_matrix: Mixing matrix.

        Returns:
        - Mixed signals.
        """
    mixed_signals = np.dot (audio_sources, mixing_matrix.T)
    return mixed_signals


# Step 5: Apply ICA
def apply_ica(mixed_signals):
    """
       Apply Independent Component Analysis (ICA) to separate mixed signals.

       Args:
       - mixed_signals: Mixed audio signals.

       Returns:
       - Reconstructed separated signals.
       """
    ica = FastICA (n_components=6, whiten="arbitrary-variance")
    reconstructed_signals = ica.fit_transform (mixed_signals)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    assert np.allclose (mixed_signals, np.dot (reconstructed_signals, A_.T) + ica.mean_)

    return reconstructed_signals


# Step 6: Normalize reconstructed signals
def normalize_signals(signals):
    """
        Normalize signals to have zero mean and unit variance.

        Args:
        - signals: Signals to be normalized.

        Returns:
        - Normalized signals.
        """
    # Normalize each signal to have zero mean and unit variance
    # normalized_signals = (signals - np.mean (signals, axis=0)) / np.std (signals, axis=0)
    separated_signals = signals - np.mean (signals, axis=0)
    separated_signals /= np.linalg.norm (separated_signals, axis=0)

    return separated_signals


# Step 7: Plot spectrogram of normalized reconstructed signals
def plot_normalized_spectrograms(normalized_signals, frame_rates):
    for i, signal in enumerate (normalized_signals.T):
        plot_signal_and_spectrogram (signal, frame_rates[i], f'Normalized Reconstructed Signal {i + 1}')


def save_wave_file(folder_path, file_name, signal_data, frame_rate):
    """
        Save signal data as a wave file.

        Args:
        - folder_path: Folder to save the file.
        - file_name: Name of the file.
        - signal_data: Signal data.
        - frame_rate: Frame rate.
        - sample_width: Sample width.
        """
    try:
        os.makedirs (folder_path, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join (folder_path, file_name)
        wavfile.write (file_path, frame_rate, signal_data)
        # print(f"Saved {file_name} successfully.")
    except Exception as e:
        print (f"Error saving wave file {file_name}: {e}")


def create_segments(signal, segment_length, overlap):
    """
    Create segments from a signal with a specified segment length and overlap.
    """
    num_samples = len (signal)
    segments = []
    start = 0
    while start + segment_length <= num_samples:
        segments.append (signal[start:start + segment_length])
        start += segment_length - overlap
    return segments


# Function to extract features from audio segment
def extract_features(segment, sr):
    # MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc (y=segment, sr=sr)
    mfcc_mean = np.mean (mfccs.T, axis=0)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid (y=segment, sr=sr)[0]
    spectral_centroid_mean = np.mean (spectral_centroid)

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth (y=segment, sr=sr)[0]
    spectral_bandwidth_mean = np.mean (spectral_bandwidth)

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast (y=segment, sr=sr, n_bands=5)[0]
    spectral_contrast_mean = np.mean (spectral_contrast)

    # Zero-crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate (segment)[0]
    zero_crossing_rate_mean = np.mean (zero_crossing_rate)
    # Root Mean Square (RMS)
    rms = librosa.feature.rms (y=segment)[0]
    rms_mean = np.mean (rms)
    spectral_rolloff = librosa.feature.spectral_rolloff (y=segment, sr=sr)[0]
    spectral_rolloff_mean = np.mean (spectral_rolloff)

    return np.concatenate ([mfcc_mean, [spectral_rolloff_mean, spectral_centroid_mean]])


# Function to segment audio
def segment_audio(audio_file, segment_length=2048, overlap=1024):
    audio, sr = librosa.load (audio_file, sr=None)
    num_samples = len (audio)
    segments = []
    start = 0
    while start + segment_length <= num_samples:
        segments.append (audio[int (start):int (start + segment_length)])  # Ensure indices are integers
        start += segment_length - overlap
    return segments, sr


# Function to segment all audio files in a directory
def segment_all_audios(audio_dir, segment_length=2048, overlap=1024):
    all_segments = []
    all_labels = []
    all_srs = []

    for file_name in os.listdir (audio_dir):
        if file_name.endswith (".wav"):  # Assuming all audio files are in .wav format
            file_path = os.path.join (audio_dir, file_name)
            segments, sr = segment_audio (file_path, segment_length, overlap)
            all_segments.extend (segments)
            # Assigning labels based on file name
            label = int (os.path.splitext (file_name)[0][
                             -1])  # Extracting label from file name (assuming last character is the label)
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


def test_classifier(classifier, scaler, X_test, y_test):
    # Scale the test features
    X_test_scaled = scaler.transform (X_test)

    # Predict labels for the test data
    y_pred = classifier.predict (X_test_scaled)

    # Evaluate the classifier
    accuracy = accuracy_score (y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Print classification report
    print (classification_report (y_test, y_pred))

    # Print confusion matrix
    print ("Confusion Matrix:")
    print (confusion_matrix (y_test, y_pred))


def main():
    # Step 1: Read wave files
    wave_files = read_wave_files_in_folder (folder_path)
    for file_name, wave_data in wave_files:
        frame_rate = wave_data['frame_rate']
        samples = wave_data['signal_data']
        # Step 2: Plot spectrogram of each signal
        plot_signal_and_spectrogram (samples, frame_rate)

    # Step 3: Generate mixing matrix
    mixing_matrix = generate_mixing_matrix ()

    # Step 4: Create mixed signals
    signals = [wave_data['signal_data'] for _, wave_data in wave_files]
    mixed_signals = np.column_stack (signals)
    mixed_signals = create_mixed_signals (mixed_signals, mixing_matrix)

    mixed_folder = os.path.join (folder_path, 'mixed')
    combined_signal = np.sum (mixed_signals.T, axis=0).astype (np.int16)
    save_wave_file (mixed_folder, f'mixed_wave.wav', combined_signal, 8000)

    for i, (file_name, wave_data) in enumerate (wave_files):
        mixed_signal = mixed_signals[:, i]  # Select the i-th column (signal) from mixed_signals
        frame_rate = wave_data['frame_rate']
        save_wave_file (mixed_folder, f'mixed_{file_name}', mixed_signal.astype (np.int16), frame_rate)

    # Step 5: Apply ICA
    reconstructed_signals = apply_ica (mixed_signals)

    # Step 6: Normalize reconstructed signals
    normalized_signals = normalize_signals (reconstructed_signals)
    normalized_folder = os.path.join (folder_path, 'normalized')
    for i, (file_name, wave_data) in enumerate (wave_files):
        normalized_signal = normalized_signals[:, i]
        save_wave_file (normalized_folder, f'normalized_{file_name}', normalized_signal,
                        wave_data['frame_rate'])

    frame_rates = [wave_data['frame_rate'] for _, wave_data in wave_files]
    # Step 7: Plot spectrogram of each normalized reconstructed signal
    plot_normalized_spectrograms (normalized_signals, frame_rates)

    scaler = StandardScaler ()

    segment_length = 5000
    overlap = 4000
    # Segment all audio files for training
    all_segments, all_labels, all_srs = segment_all_audios (train_audio_directory, segment_length=segment_length, overlap=overlap)

    # Convert segments to feature vectors
    X = np.array ([extract_features (segment, sr) for segment, sr in zip (all_segments, all_srs)])
    y = np.array (all_labels)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform (X_train)

    # Train SVM classifier
    svm_model = train_svm (X_train_scaled, y_train)
    test_classifier (svm_model, scaler, X_test, y_test)

    # Predict labels for new audio files
    predictions = predict_new_audios (normalized_audio_directory, svm_model, segment_length=segment_length, overlap=overlap)

    # Print predictions for each new audio file
    for file_name, prediction in predictions.items ():
        print (f"Audio file: {file_name}, Predicted Label: {prediction}")


    # Predict labels for new audio files
    predictions = predict_new_audios (train_audio_directory, svm_model, segment_length=segment_length, overlap=overlap)

    # Print predictions for each new audio file
    for file_name, prediction in predictions.items ():
        print (f"Audio file: {file_name}, Predicted Label: {prediction}")


if __name__ == "__main__":
    main ()
