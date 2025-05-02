"""
Functions to preprocess the .mp3 files into mel-spectrograms.
"""

# At the top of src/preprocess.py

import pandas as pd
import numpy as np
import librosa
import os
import ast  # To parse the multi_hot_label string if needed
import logging
from tqdm import tqdm # For progress bar: pip install tqdm

import sys

cwd = os.getcwd()
print(f'cwd: {cwd}')
PROJECT_ROOT = os.path.abspath(os.path.join(cwd, '../')) # NOTE: remember to change if change the directory structure

print(f'PROJECT_ROOT: {PROJECT_ROOT}')


# Add project root to Python's module search path
sys.path.append(PROJECT_ROOT)
import src.utils as utils  
from src.utils import get_audio_path as get_audio_path 
import config # NOTE: need to check config to make sure all paths and parameters are correct


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Helper Function (or import from utils.py) ---
# It's cleaner to put this in src/utils.py


# this get_audio_path is exactly the same as the one in utils.py, I just copy it here to make understanding easier
def get_audio_path(audio_dir, track_id):
    """Constructs the path to an FMA audio file."""
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def extract_mel_spectrogram(audio_path, params):
    """Loads audio, computes Log-Mel Spectrogram using parameters from config."""
    try:
        # 1. Pull all preprocessing settings out of the params dict
        sr                  = params["sample_rate"]              
        duration            = params["segment_duration_seconds"] 
        samples_per_segment = params["samples_per_segment"]      
        n_fft               = params["n_fft"]                    
        hop_length          = params["hop_length"]               
        n_mels              = params["n_mels"]                   

        # 2. Load the audio file
        #    - resamples to `sr`,  
        #    - only reads up to `duration` seconds
        y, loaded_sr = librosa.load(
            audio_path,
            sr=sr,
            duration=duration
        )

        # 3. Pad if shorter than expected duration (librosa pads by default, but explicit is safer)
        if len(y) < samples_per_segment:
            # 3a. If too short, pad the end with zeros
            y = np.pad(y, (0, samples_per_segment - len(y)))
        elif len(y) > samples_per_segment:
            # 3b. If too long (rare, but just in case), truncate
            y = y[:samples_per_segment]

        # 4. Compute the Mel spectrogram
        melspectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # 5. Convert power to decibel (log scale), normalizing by the max power
        log_melspectrogram = librosa.power_to_db(
            melspectrogram,
            ref=np.max
        )

        # 6. Return the final 2D array (shape: [n_mels, time_frames])
        return log_melspectrogram

    except Exception as e:
        # 7. If anything fails (file missing, decode error, etc.), log a warning...
        logging.warning(
            f"Could not process file {os.path.basename(audio_path)}: {e}"
        )
        # 8. ...and return None so downstream code can skip it gracefully
        return None



# --- Step 2: Main Processing Logic ---

def preprocess_audio_features():
    """
    Extracts log-mel spectrogram features for each track in the small FMA subset
    and writes them to disk, plus builds a CSV manifest linking features to labels.

    Side effects (outputs on disk):
      1. A directory of `.npy` files, one per track, in `config.FMA_FEATURES_DIR`.
         - Filenames are `<track_id>.npy`
         - Each file contains a 2D NumPy array of shape (n_mels, time_frames).
      2. A CSV file named `final_feature_manifest.csv` in `config.PROCESSED_DATA_DIR`.
         - Columns:
           • track_id      – integer ID of the track
           • feature_path  – relative path to the saved `.npy` file
           • label_vector  – the multi-hot list of genre labels
           • split         – which dataset split the track belongs to (train/val/test)

    Mock up example of the manifest file:
    | track_id | feature_path              | label_vector         | split |
    |---------:|---------------------------|----------------------|:-----:|
    |   123456 | data/features/123456.npy  | [0,1,0,0,1,…]        | train |
    |   234567 | data/features/234567.npy  | [1,0,0,1,0,…]        | test  |
    |    …     | …                         | …                    |  …    |



    Workflow:
      1. Load `small_subset_multihot.csv` (must exist in `PROCESSED_DATA_DIR`).
      2. Parse the `multi_hot_label` strings back into Python lists.
      3. Ensure output directory for features exists.
      4. Loop over each track:
         a. Check audio file path is valid.
         b. Compute log-mel spectrogram via `extract_mel_spectrogram`.
         c. If successful, save the array as `<track_id>.npy` and record its info.
      5. After the loop, save all recorded entries into `final_feature_manifest.csv`.

    Logging:
      - INFO for major milestones (start/end, counts).
      - WARNING for skipped tracks (missing files or failed extraction).
      - ERROR for unrecoverable issues (failed file I/O or metadata parsing).

    Returns:
      None
    """


    logging.info("--- Starting Feature Extraction ---")

    # --- Load the final metadata with multi-hot labels ---
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'small_subset_multihot.csv')
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}. Run label processing first.")
        return

    logging.info(f"Loading metadata with multi-hot labels from {metadata_path}")
    try:
        # Load the CSV, keeping track_id as index
        metadata_df = pd.read_csv(metadata_path, index_col='track_id')

        # --- CRITICAL: Parse the 'multi_hot_label' string back into a list/array ---
        # The .to_csv likely saved the list as its string representation.
        # We use ast.literal_eval to safely convert it back.
        metadata_df['multi_hot_label'] = metadata_df['multi_hot_label'].apply(ast.literal_eval)
        
        
        # Considering converting inner list to numpy array (if needed downstream), keep as list is fine too. so if want to convert, uncomment the following line.
        # metadata_df['multi_hot_label'] = metadata_df['multi_hot_label'].apply(np.array)

        logging.info(f"Loaded and parsed metadata for {len(metadata_df)} tracks.")
        logging.info("Example parsed label vector:")
        # Display first element to check format
        print(metadata_df['multi_hot_label'].iloc[0])

    except Exception as e:
        logging.error(f"Failed to load or parse metadata CSV '{metadata_path}': {e}", exc_info=True)
        return

    # --- Create output directory for features ---
    features_output_dir = config.FMA_FEATURES_DIR
    os.makedirs(features_output_dir, exist_ok=True)
    logging.info(f"Ensured processed features directory exists: {features_output_dir}")

    # --- Loop, Extract, Save ---
    manifest_data = [] # To store info for the final manifest file
    processed_count = 0
    error_count = 0

    logging.info(f"Starting feature extraction loop for {len(metadata_df)} tracks...")
    # Use tqdm for a progress bar in the terminal
    for track_id, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        audio_path = row['audio_path']
        multi_hot_label = row['multi_hot_label'] # Now it's a list/array
        split = row['split']

        # Final check if audio path really exists
        if not isinstance(audio_path, str) or not os.path.exists(audio_path):
             logging.warning(f"Audio path missing or invalid for track {track_id}: '{audio_path}'. Skipping.")
             error_count += 1
             continue

        # Extract features using the function defined earlier, this is the .npy file content
        log_melspec = extract_mel_spectrogram(audio_path, config.PREPROCESSING_PARAMS)

        if log_melspec is not None:
            # Define path to save the feature file (.npy)
            feature_filename = f"{track_id}.npy"
            # Store relative path from project root in manifest for portability
            relative_feature_path = os.path.join(os.path.relpath(features_output_dir, PROJECT_ROOT), feature_filename)
            absolute_feature_path = os.path.join(features_output_dir, feature_filename)

            # Save the feature array
            try:
                np.save(absolute_feature_path, log_melspec)

                # Add entry to manifest list
                manifest_data.append({
                    'track_id': track_id,
                    'feature_path': relative_feature_path, # Use relative path
                    'label_vector': multi_hot_label,       # Store the actual list/vector
                    'split': split
                })
                processed_count += 1
            except Exception as e:
                logging.error(f"Failed to save feature for track {track_id} to {absolute_feature_path}: {e}")
                error_count += 1
        else:
            # Feature extraction failed (error already logged in the function)
            error_count += 1

    logging.info("Feature extraction loop finished.")
    logging.info(f"Successfully processed and saved features for: {processed_count} tracks.")
    logging.info(f"Errors/Skipped during feature extraction: {error_count} tracks.")

    # --- Save the final manifest ---
    if not manifest_data:
        logging.warning("No features were successfully processed. Manifest file will be empty.")
        return

    manifest_df = pd.DataFrame(manifest_data)
    # Reorder columns for clarity
    manifest_df = manifest_df[['track_id', 'feature_path', 'label_vector', 'split']]

    manifest_path = os.path.join(config.PROCESSED_DATA_DIR, 'final_feature_manifest.csv')
    try:
        # Save the manifest mapping track IDs to feature paths and labels
        manifest_df.to_csv(manifest_path, index=False)
        logging.info(f"Saved final manifest file ({len(manifest_df)} entries) to {manifest_path}")
    except Exception as e:
        logging.error(f"Failed to save manifest file: {e}", exc_info=True)


# --- Make the script runnable ---
if __name__ == '__main__':
    # This executes when you run `python src/preprocess.py` from the terminal
    # It will now load the multi-hot metadata and process the audio features.
    preprocess_audio_features()