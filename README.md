# Music Generation Project

This project focuses on music generation using machine learning techniques.

## Project Structure
- `data/`: Contains datasets
  - `raw/`: Raw, unprocessed music data
    - `fma_metadata/`: FMA dataset metadata (tracks.csv, genres.csv, etc.)
    - `fma_audio/`: FMA audio files
  - `processed/`: Cleaned and preprocessed data ready for modeling
    - `fma_features/`: Extracted features from audio files
- `docs/`: Documentation and references
- `models/`: Saved model checkpoints
- `notebooks/`: Jupyter notebooks for exploration and analysis
  - `exploratory.ipynb`: Initial data exploration
- `src/`: Source code for the project
  - `models/`: Code for model definitions and training
    - `train_model.py`: Script for training models
    - `evaluate_model.py`: Script for evaluating models
  - `visualization/`: Code for data and results visualization
  - `make_dataset.py`: Script to download or generate the dataset
  - `preprocess.py`: Script to process data from raw to processed
  - `utils.py`: Utility functions used across the project
- `tests/`: Unit tests for code validation
- `.gitignore`: Specifies intentionally untracked files
- `config.py`: Configuration settings (paths, parameters)
- `main.py`: Main script to run the full pipeline
- `requirements.txt`: Project dependencies

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Download the dataset: `python src/make_dataset.py`
3. Preprocess the data: `python src/preprocess.py`
4. Explore the data: Open `notebooks/exploratory.ipynb`
5. Train a model: `python src/models/train_model.py`
6. Evaluate the model: `python src/models/evaluate_model.py`

Alternatively, run the full pipeline with: `python main.py` 