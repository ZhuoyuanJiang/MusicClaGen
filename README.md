# FMA Dataset Processing and Music Genre Classification

This repository contains code for processing the FMA (Free Music Archive) dataset and training music genre classification models using deep learning.

## Project Overview

This project focuses on music genre classification using the FMA (Free Music Archive) dataset. We process the dataset, extract audio features, and train deep learning models (specifically Wav2VecBERT) to classify music genres. The implementation includes a complete pipeline from data preprocessing to model training and evaluation.

## Dataset

We use the FMA (Free Music Archive) dataset, which is a publicly available dataset for music analysis:

- The dataset contains Creative Commons-licensed audio from thousands of artists
- We specifically use the FMA Small subset, which includes 8,000 30-second clips of songs from 8 different genres
- The dataset provides audio files along with rich metadata including genre labels

For more information about the dataset, visit the [official FMA GitHub repository](https://github.com/mdeff/fma).

## Installation

### Prerequisites

- Python 3.8+
- 7-Zip for extracting compressed files
- Required Python libraries (see `environment.yml`)
- Conda for creating the environment 

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install required Python packages:
```bash
conda env create -f environment.yml
```

Manually install PyTorch and Torchvision:
```
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

If this line fails, we used PyTorch 2.1.0, torchvision 0.16.0, and torchaudio 2.1.0, all built with CUDA 11.8 support.


3. Download 7-Zip from [7-zip.org](https://www.7-zip.org/)

## Data Preparation

Follow these steps to prepare the FMA dataset:

1. Download the metadata package:
   - Go to [FMA GitHub repository](https://github.com/mdeff/fma)
   - Download `fma_metadata.zip` from the README.md DATA section

2. Extract `fma_metadata.zip` using 7-Zip

3. Place the extracted `fma_metadata` folder in `data/raw/`

4. Download `fma_small.zip` from the FMA GitHub repository

5. Extract `fma_small.zip` and place the extracted folder in `data/raw/fma_audio/`

6. Run the first part of `src/data/raw/usage.ipynb` to generate `small_subset.csv`

7. Run the entire 1.2 section of the notebook to generate:
   - `small_subset_multihot.csv`
   - `unified_genres.txt`

   **Note**: Each user must run this notebook individually as the file paths in `small_subset_multihot.csv` are system-specific.

8. Run the preprocessing script:

src/data/melspecto_conversion_exploration.ipynb

This will generate:
- 7994/8000 `.npy` feature files in `data/processed/fma_features/fma_small/`
- `final_feature_manifest.csv` in `data/processed/`

## File Structure

```
.
├── data/
│   ├── raw/
│   │   ├── fma_metadata/       # Extracted metadata
│   │   └── fma_audio/
│   │       └── fma_small/      # Audio files
│   └── processed/
│       ├── fma_features/
│       │   └── fma_small/      # Extracted features
│       └── final_feature_manifest.csv
│       └── small_subset_multihot.csv
│       └── small_subset.csv
│       └── unified_genres.txt
├── src/
│   ├── data/
│   │   └── raw/
│   │       └── usage.ipynb    # Data processing notebook
│   ├── models/
│   │   └── train_classification_model_exploration_Wav2Vecbert2 copy.ipynb  # Training notebook
│   └── melspecto_conversion_exploration.ipynb         # Feature extraction
├── environment2.yml
└── README.md
```

## Data Processing Pipeline

1. **Data Download & Organization**: Download and extract the FMA dataset files
2. **Metadata Preparation**: Process metadata to create CSV files with multi-hot genre encodings
3. **Feature Extraction**: Extract audio features and save them as .npy files
4. **Manifest Creation**: Generate a manifest file linking features to genre labels

## Model Training

To train the model:

1. Open `src/models/train_classification_model_exploration_Wav2Vecbert2 copy.ipynb`  (NOTE: copy here is important!)
2. Run all cells in the notebook

This notebook:
- Loads the raw audio files and turn them into mel-spectrogram
- Implements a deep learning model based on Wav2VecBERT architecture
- Trains the model for music genre classification
- Evaluates model performance
- Saves the trained model

## Known Issues

Refer to the [FMA GitHub wiki](https://github.com/mdeff/fma/wiki) for known issues with the dataset and how to handle them.

## References

- [FMA Dataset GitHub Repository](https://github.com/mdeff/fma)
- Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. 18th International Society for Music Information Retrieval Conference (ISMIR).

## License

- The code in this repository is released under the MIT license
- The FMA metadata is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
- The audio files are distributed under the terms of the license chosen by the artists (mostly Creative Commons)