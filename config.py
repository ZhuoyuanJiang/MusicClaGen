"""
Configuration settings for the music generation project.
"""
import os

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to project root
PATHS = {
    'PROJECT_ROOT': PROJECT_ROOT,
    'DATA_DIR': os.path.join(PROJECT_ROOT, 'data'),
    'RAW_DATA_DIR': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'PROCESSED_DATA_DIR': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'AUDIO_DIR': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_audio', 'fma_small'),
    'METADATA_DIR': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_metadata'),
    'TRACKS_PATH': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_metadata', 'tracks.csv'),
    'GENRES_PATH': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_metadata', 'genres.csv'),
    'FEATURES_PATH': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_metadata', 'features.csv'),
    'ECHONEST_PATH': os.path.join(PROJECT_ROOT, 'data', 'raw', 'fma_metadata', 'echonest.csv'),
    'MODELS_DIR': os.path.join(PROJECT_ROOT, 'models'),
    'FIGURES_DIR': os.path.join(PROJECT_ROOT, 'figures'),
    'FMA_FEATURES_DIR': os.path.join(PROJECT_ROOT, 'data', 'processed', 'fma_features', 'fma_small'),
}

# Data processing parameters
PREPROCESSING_PARAMS = {
    "sample_rate": 16000,   # sampling rate in Hertz, Common values are 16000 (speech, some audio models), 22050 (common for music ML), 44100 (CD quality).
    # We use 16000 because we want to match our pre-trained model here. But note that the original FMA dataset used 44100. 
    "n_fft": 1024,          # window length (in samples) for each FFT
    "hop_length": 256,      # step size (in samples) between successive frames
    "n_mels": 128,          # how many Mel bands to generate
    "segment_duration_seconds": 30  # Duration of audio segments to process
}

# Calculate samples per segment based on duration and sample rate
PREPROCESSING_PARAMS["samples_per_segment"] = int(
    PREPROCESSING_PARAMS["sample_rate"] * PREPROCESSING_PARAMS["segment_duration_seconds"]
)

# Model parameters
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "hidden_dim": 128
} 

# Function to ensure directories exist
def ensure_directories_exist():
    """Create directories if they don't exist"""
    for key, path in PATHS.items():
        if key.endswith('_DIR') and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")




print(PROJECT_ROOT)