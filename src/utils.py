"""
Utility functions for the project.
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_audio_file(file_path):
    """
    Load an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    # Placeholder - replace with actual audio loading code
    # Example:
    # import librosa
    # audio_data, sample_rate = librosa.load(file_path, sr=None)
    # return audio_data, sample_rate
    
    logger.info(f"Loading audio file: {file_path}")
    return np.array([]), 44100  # Placeholder 