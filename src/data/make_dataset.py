"""
Script to download or generate the dataset.
"""
import os
import argparse
import logging
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def main(input_filepath=None, output_filepath=None):
    """
    Main function to create or download the dataset.
    
    Args:
        input_filepath (str, optional): Path to input data. Defaults to config.DATA_RAW_DIR
        output_filepath (str, optional): Path to save processed data. Defaults to config.DATA_PROCESSED_DIR
    """
    # Use default paths from config if not provided
    if input_filepath is None:
        input_filepath = DATA_RAW_DIR
    if output_filepath is None:
        output_filepath = DATA_PROCESSED_DIR
    
    logger = logging.getLogger(__name__)
    logger.info(f'Making dataset from raw data in {input_filepath}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Add your dataset creation code here
    # Example:
    # 1. List all files in the raw directory
    raw_files = [f for f in os.listdir(input_filepath) if not f.startswith('.')]
    logger.info(f'Found {len(raw_files)} files in raw data directory')
    
    # 2. Process each file (placeholder)
    for file in raw_files:
        logger.info(f'Processing {file}')
        # Your processing code here
    
    logger.info(f'Dataset saved to {output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser(description='Create dataset from raw data')
    parser.add_argument('--input_filepath', type=str, help='Path to raw data', default=None)
    parser.add_argument('--output_filepath', type=str, help='Path to save processed data', default=None)
    
    args = parser.parse_args()
    main(args.input_filepath, args.output_filepath) 