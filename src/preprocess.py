"""
Script to preprocess the raw data.
"""
import os
import argparse
import logging
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PREPROCESSING_PARAMS

def preprocess_data(input_filepath=None, output_filepath=None):
    """
    Preprocess the raw data.
    
    Args:
        input_filepath (str, optional): Path to raw data. Defaults to config.DATA_RAW_DIR
        output_filepath (str, optional): Path to save processed data. Defaults to config.DATA_PROCESSED_DIR
    """
    # Use default paths from config if not provided
    if input_filepath is None:
        input_filepath = DATA_RAW_DIR
    if output_filepath is None:
        output_filepath = DATA_PROCESSED_DIR
    
    logger = logging.getLogger(__name__)
    logger.info(f'Preprocessing data from {input_filepath}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Add your preprocessing code here
    # Example for FMA dataset:
    # 1. Load metadata
    logger.info("Loading metadata...")
    # metadata = load_metadata(os.path.join(input_filepath, 'fma_metadata'))
    
    # 2. Process audio files and extract features
    logger.info("Extracting features from audio files...")
    # features = extract_features(os.path.join(input_filepath, 'fma_audio'), metadata, PREPROCESSING_PARAMS)
    
    # 3. Save processed data
    logger.info("Saving processed data...")
    # save_features(features, os.path.join(output_filepath, 'fma_features'))
    
    logger.info(f'Processed data saved to {output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser(description='Preprocess raw data')
    parser.add_argument('--input_filepath', type=str, help='Path to raw data', default=None)
    parser.add_argument('--output_filepath', type=str, help='Path to save processed data', default=None)
    
    args = parser.parse_args()
    preprocess_data(args.input_filepath, args.output_filepath) 