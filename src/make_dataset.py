"""
Script to download or generate the dataset.
"""
import os
import argparse
import logging
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RAW_DIR

def main(output_filepath=None):
    """
    Main function to download or create the dataset.
    
    Args:
        output_filepath (str, optional): Path to save the raw data. Defaults to config.DATA_RAW_DIR
    """
    # Use default path from config if not provided
    if output_filepath is None:
        output_filepath = DATA_RAW_DIR
    
    logger = logging.getLogger(__name__)
    logger.info(f'Downloading/creating dataset to {output_filepath}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Add your dataset download/creation code here
    # Example for FMA dataset:
    # 1. Download metadata
    logger.info("Downloading FMA metadata...")
    # download_fma_metadata(os.path.join(output_filepath, 'fma_metadata'))
    
    # 2. Download audio files
    logger.info("Downloading FMA audio files...")
    # download_fma_audio(os.path.join(output_filepath, 'fma_audio'))
    
    logger.info(f'Dataset saved to {output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser(description='Download or create the dataset')
    parser.add_argument('--output_filepath', type=str, help='Path to save raw data', default=None)
    
    args = parser.parse_args()
    main(args.output_filepath) 