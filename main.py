"""
Main script to run the full music generation pipeline.
"""
import os
import argparse
import logging
from src.make_dataset import main as make_dataset
from src.preprocess import preprocess_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR

def run_pipeline(raw_data_path=DATA_RAW_DIR, 
                processed_data_path=DATA_PROCESSED_DIR,
                model_path=os.path.join(MODELS_DIR, "music_model.pkl"),
                skip_download=False,
                skip_preprocessing=False,
                skip_training=False,
                skip_evaluation=False):
    """
    Run the full music generation pipeline.
    
    Args:
        raw_data_path (str): Path to raw data
        processed_data_path (str): Path to save processed data
        model_path (str): Path to save the trained model
        skip_download (bool): Skip the data download step
        skip_preprocessing (bool): Skip the preprocessing step
        skip_training (bool): Skip the model training step
        skip_evaluation (bool): Skip the model evaluation step
    """
    logger = logging.getLogger(__name__)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Step 1: Download/create dataset
    if not skip_download:
        logger.info("Step 1: Downloading/creating dataset")
        make_dataset(raw_data_path)
    else:
        logger.info("Skipping dataset download")
    
    # Step 2: Preprocess data
    if not skip_preprocessing:
        logger.info("Step 2: Preprocessing data")
        preprocess_data(raw_data_path, processed_data_path)
    else:
        logger.info("Skipping preprocessing")
    
    # Step 3: Train model
    if not skip_training:
        logger.info("Step 3: Training model")
        train_model(processed_data_path, model_path)
    else:
        logger.info("Skipping model training")
    
    # Step 4: Evaluate model
    if not skip_evaluation:
        logger.info("Step 4: Evaluating model")
        metrics = evaluate_model(model_path, processed_data_path)
    else:
        logger.info("Skipping model evaluation")
        metrics = {}
    
    logger.info("Pipeline completed successfully!")
    return metrics

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser(description='Run the music generation pipeline')
    parser.add_argument('--raw_data_path', type=str, default=DATA_RAW_DIR, 
                        help='Path to raw data')
    parser.add_argument('--processed_data_path', type=str, default=DATA_PROCESSED_DIR, 
                        help='Path to save processed data')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODELS_DIR, "music_model.pkl"), 
                        help='Path to save the model')
    parser.add_argument('--skip_download', action='store_true', help='Skip data download step')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip model evaluation step')
    
    args = parser.parse_args()
    run_pipeline(
        args.raw_data_path, 
        args.processed_data_path, 
        args.model_path,
        args.skip_download,
        args.skip_preprocessing,
        args.skip_training,
        args.skip_evaluation
    ) 