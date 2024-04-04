# Audio Source Separation and Classification

## Overview

This project explores the use of Independent Component Analysis (ICA) for separating mixed audio signals and then classifying the separated signals using a machine learning classification algorithm. The goal is to separate mixed audio sources, normalize the reconstructed signals, and then classify them accurately based on their content.

## Features

- Read wave files and extract relevant information.
- Plot spectrogram of audio signals.
- Generate random mixing matrix.
- Create mixed signals by applying the mixing matrix.
- Apply ICA to separate mixed signals into reconstructed signals.
- Normalize reconstructed signals.
- Plot spectrogram of normalized reconstructed signals.
- Segment audio files and extract features for classification.
- Train a Support Vector Machine (SVM) classifier.
- Test classifier performance on original and separated audio signals.
- Predict labels for new audio files using the trained classifier.

## Requirements

- Python 3.x
- NumPy
- SciPy
- scikit-learn
- matplotlib
- librosa

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/audio-separation-classification.git
cd audio-separation-classification
