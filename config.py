"""
Configuration settings for the music generation project.
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# FMA specific paths
FMA_METADATA_DIR = os.path.join(DATA_RAW_DIR, "fma_metadata")
FMA_AUDIO_DIR = os.path.join(DATA_RAW_DIR, "fma_audio")
FMA_FEATURES_DIR = os.path.join(DATA_PROCESSED_DIR, "fma_features")

# Model parameters
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "hidden_dim": 128
}

# Data processing parameters
PREPROCESSING_PARAMS = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128
} 