# src/models/train_classification_wav2vec2bert.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor, # Use Feature Extractor for Wav2Vec2-BERT based on our findings
    AdamW,
    get_linear_schedule_with_warmup # Optional: for learning rate scheduling
)
import pandas as pd
import numpy as np
import os
import sys
import ast
import re # If using regex parser for labels
import logging
import time
from sklearn.metrics import hamming_loss, jaccard_score, f1_score
from tqdm import tqdm
import librosa

# --- Project Setup ---
try:
    # Assumes this script is in src/models/, adjust relative path if needed
    cwd = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(cwd, '../../'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively sometimes)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '../..')) # Adjust as needed
    if PROJECT_ROOT not in sys.path:
         sys.path.append(PROJECT_ROOT)
    print(f"PROJECT_ROOT set using cwd fallback: {PROJECT_ROOT}")


try:
    import config # Import your configuration file
except ModuleNotFoundError:
    print("ERROR: Cannot import config.py. Make sure PROJECT_ROOT is correct and in sys.path.")
    sys.exit(1)

# --- Setup Logging ---
# Clear previous handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to console

# --- Optional: Label String Parser (if needed) ---
def parse_numpy_array_string(array_str):
    """ Parses strings like '[np.float32(1.0), ...]' into list of 0s/1s """
    if not isinstance(array_str, str): return []
    try:
        float_matches = re.findall(r'np\.float32\((\d+\.\d+)\)', array_str)
        values = []
        for match in float_matches:
            value = float(match)
            values.append(1 if value == 1.0 else 0 if value == 0.0 else value) # Convert 1.0/0.0 to int
        return values
    except Exception as e:
        logging.warning(f"Error parsing array string: {e}")
        return []

# --- Dataset Class Definition ---
class FMARawAudioDataset(Dataset):
    """ Loads raw audio waveforms and labels from manifest, uses HF feature extractor. """
    def __init__(self, manifest_path, feature_extractor):
        logging.info(f"Initializing FMARawAudioDataset from: {manifest_path}")
        if feature_extractor is None:
            raise ValueError("Dataset requires a feature_extractor instance.")
        self.feature_extractor = feature_extractor
        try:
            self.target_sr = self.feature_extractor.sampling_rate
            logging.info(f"Target sampling rate set from feature extractor: {self.target_sr} Hz")
        except AttributeError:
            logging.error("Could not get sampling_rate from feature_extractor.", exc_info=True)
            raise
        try:
            self.manifest = pd.read_csv(manifest_path)
            if 'track_id' in self.manifest.columns:
                self.manifest = self.manifest.set_index('track_id', drop=False)
            # --- Choose Correct Label Parser ---
            # label_parser = parse_numpy_array_string # Use if labels look like '[np.float32(1.0)...]'
            label_parser = ast.literal_eval       # Use if labels look like '[1.0, 0.0,...]'
            # ----------------------------------
            self.manifest['multi_hot_label'] = self.manifest['multi_hot_label'].apply(label_parser)
            logging.info(f"Loaded and parsed manifest with {len(self.manifest)} entries.")
        except Exception as e:
            logging.error(f"Error loading/parsing manifest {manifest_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.manifest.iloc[idx]
        track_id = row.get('track_id', self.manifest.index[idx])
        label_vector = row['multi_hot_label']
        audio_path = row['audio_path']
        if not os.path.isabs(audio_path):
             audio_path = os.path.join(config.PROJECT_ROOT, audio_path)

        try:
            waveform, loaded_sr = librosa.load(audio_path, sr=self.target_sr, duration=30.0)
            if len(waveform) < int(0.1 * self.target_sr):
                logging.warning(f"Audio signal for track {track_id} too short, returning None.")
                return None # Needs collate_fn that handles None or filter these out earlier

            # Apply Feature Extractor
            inputs = self.feature_extractor(
                waveform, sampling_rate=self.target_sr, return_tensors="pt",
                padding="max_length", truncation=True, max_length=5000, # Use max_length from model config
                return_attention_mask=True
            )

            feature_tensor = inputs.get('input_values', inputs.get('input_features'))
            if feature_tensor is None: raise KeyError("Input key not found.")
            feature_tensor = feature_tensor.squeeze(0)

            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None: attention_mask = attention_mask.squeeze(0)

            label_tensor = torch.tensor(label_vector, dtype=torch.float32)

            model_input_dict = {"labels": label_tensor}
            if 'input_values' in inputs: model_input_dict['input_values'] = feature_tensor
            elif 'input_features' in inputs: model_input_dict['input_features'] = feature_tensor
            if attention_mask is not None: model_input_dict['attention_mask'] = attention_mask
            return model_input_dict

        except FileNotFoundError:
             logging.error(f"Audio file not found for track {track_id} at {audio_path}")
             return None # Needs collate_fn that handles None
        except Exception as e:
            logging.error(f"Error loading/processing track {track_id} at {audio_path}: {e}", exc_info=True)
            return None # Needs collate_fn that handles None

# --- Data Collator Definition ---
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class DataCollatorAudio:
    padding_value: float = 0.0
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Filter out None entries if Dataset returns None on error
        valid_features = [f for f in features if f is not None]
        if not valid_features:
             # If all samples in batch failed, return empty or handle appropriately
             logging.warning("Collate function received empty batch after filtering Nones.")
             return {} # Or raise error, depending on desired behavior

        input_key = 'input_values' if 'input_values' in valid_features[0] else 'input_features'
        input_features = [d[input_key] for d in valid_features]

        # Pad features (assuming shape [SeqLen, FeatureDim] or [FeatureDim, SeqLen])
        # Determine sequence length dimension
        if len(input_features[0].shape) == 2:
             # Assume last dim is sequence if feature dim is smaller, else first dim
             seq_len_dim = -1 if input_features[0].shape[0] > input_features[0].shape[1] else 0
        elif len(input_features[0].shape) == 1:
             seq_len_dim = 0 # Assume 1D tensor is sequence
        else:
             # Default or raise error for unexpected shapes
             logging.warning(f"Unexpected feature tensor shape {input_features[0].shape}, assuming last dim is sequence.")
             seq_len_dim = -1

        max_len = max(feat.shape[seq_len_dim] for feat in input_features)

        padded_features = []
        for feat in input_features:
            pad_width = max_len - feat.shape[seq_len_dim]
            # Create padding tuple - pad only the sequence dimension
            # Example for shape [SeqLen, FeatDim] (seq_len_dim=0): (0, 0, 0, pad_width) -> pad dim 0 after
            # Example for shape [FeatDim, SeqLen] (seq_len_dim=-1 or 1): (0, pad_width) -> pad dim -1 after
            if seq_len_dim == 0 and len(feat.shape)==2: # [SeqLen, FeatDim]
                 padding = (0, 0, 0, pad_width)
            elif seq_len_dim == -1 and len(feat.shape)==2: # [FeatDim, SeqLen]
                 padding = (0, pad_width)
            elif len(feat.shape)==1: # [SeqLen]
                 padding = (0, pad_width)
            else: # Fallback/Guess
                  padding = (0, pad_width) # Assumes sequence is last dim

            padded_feat = torch.nn.functional.pad(feat, padding, mode='constant', value=self.padding_value)
            padded_features.append(padded_feat)

        batch_input_features = torch.stack(padded_features)
        batch = {input_key: batch_input_features}

        # Pad attention mask if present
        if "attention_mask" in valid_features[0] and valid_features[0]["attention_mask"] is not None:
            attention_masks = [d["attention_mask"] for d in valid_features]
            max_mask_len = max(m.shape[-1] for m in attention_masks) # Mask is usually [SeqLen]
            padded_masks = []
            for mask in attention_masks:
                 pad_width = max_mask_len - mask.shape[-1]
                 padded_mask = torch.nn.functional.pad(mask, (0, pad_width), mode='constant', value=0)
                 padded_masks.append(padded_mask)
            batch["attention_mask"] = torch.stack(padded_masks)

        # Stack Labels
        labels = [d["labels"] for d in valid_features]
        batch["labels"] = torch.stack(labels)
        return batch

# --- Metrics Function ---
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    probs = 1 / (1 + np.exp(-logits_np)); preds = (probs > 0.5).astype(int); labels_np = labels_np.astype(int)
    if labels_np.shape != preds.shape: return {'hamming_loss': 1.0, 'jaccard_samples': 0.0, 'f1_micro': 0.0, 'f1_macro': 0.0}
    metrics = {}
    try:
        metrics['hamming_loss'] = hamming_loss(labels_np, preds)
        metrics['jaccard_samples'] = jaccard_score(labels_np, preds, average='samples', zero_division=0)
        metrics['f1_micro'] = f1_score(labels_np, preds, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels_np, preds, average='macro', zero_division=0)
    except Exception as e: logging.error(f"Metric calculation error: {e}")
    return metrics

# --- Training Epoch Function ---
def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps, scheduler=None): # Add scheduler
    model.train(); total_loss = 0; num_samples = 0; optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(progress_bar):
        if batch is None or not batch: continue # Skip None batches if collate_fn filters them
        try:
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**model_inputs)
            logits = outputs.logits; loss = criterion(logits, labels)
            if torch.isnan(loss): logging.warning(f"NaN loss @ step {step}"); continue
            scaled_loss = loss / gradient_accumulation_steps; scaled_loss.backward()
            batch_size_actual = labels.size(0); total_loss += loss.item() * batch_size_actual; num_samples += batch_size_actual
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional clipping
                optimizer.step()
                if scheduler: scheduler.step() # Step scheduler with optimizer
                optimizer.zero_grad()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        except Exception as e: logging.error(f"Training step {step} error: {e}", exc_info=True); continue
    if num_samples == 0: logging.warning("No samples processed in training epoch."); return 0.0
    avg_loss = total_loss / num_samples
    logging.info(f"Average Training Loss for Epoch: {avg_loss:.4f}")
    return avg_loss

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device):
    model.eval(); total_loss = 0; all_logits = []; all_labels = []; num_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None or not batch: continue
            try:
                model_inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**model_inputs); logits = outputs.logits; loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0); num_samples += labels.size(0)
                all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
            except Exception as e: logging.error(f"Evaluation batch error: {e}", exc_info=True); continue
    if not all_logits or not all_labels or num_samples == 0: logging.warning("Eval yielded no results."); return {}
    avg_loss = total_loss / num_samples
    all_logits_cat = torch.cat(all_logits, dim=0); all_labels_cat = torch.cat(all_labels, dim=0)
    eval_preds = (all_logits_cat, all_labels_cat); metrics = compute_metrics(eval_preds)
    metrics['eval_loss'] = avg_loss
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    for name, value in metrics.items():
        if name != 'eval_loss': logging.info(f"  Validation {name.replace('_', ' ').title()}: {value:.4f}")
    return metrics

# --- DataLoader Setup Function ---
def setup_dataloaders(feature_extractor):
    logging.info("--- Setting up DataLoaders ---")
    manifest_path = config.PATHS['SMALL_MULTILABEL_PATH'] # Use correct manifest key
    full_dataset = FMARawAudioDataset(manifest_path, feature_extractor=feature_extractor)
    manifest_df = full_dataset.manifest
    train_indices = manifest_df[manifest_df['split'] == 'training'].index.tolist()
    val_indices = manifest_df[manifest_df['split'] == 'validation'].index.tolist()
    test_indices = manifest_df[manifest_df['split'] == 'test'].index.tolist()
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    logging.info(f"Using FULL Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    data_collator = DataCollatorAudio()
    batch_size = config.MODEL_PARAMS["batch_size"]
    # Use num_workers > 0 on Linux/Mac for performance, 0 on Windows generally
    num_workers = 4 if os.name == 'posix' else 0
    pin_memory = True if config.DEVICE == 'cuda' else False

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers, pin_memory=pin_memory)
    logging.info("DataLoaders created using FULL splits.")
    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) # Return train size for scheduler

# --- Main Training Function ---
def run_training():
    logging.info("--- Starting Training Process ---")
    try:
        with open(config.PATHS['GENRE_LIST_PATH'], 'r') as f:
            unified_genres = [line.strip() for line in f if line.strip()]
        num_labels = len(unified_genres)
        if num_labels == 0: raise ValueError("Unified genre list empty.")
        logging.info(f"Loaded {num_labels} unified genres.")
    except Exception as e: logging.error(f"Failed load genre list: {e}",exc_info=True); sys.exit(1)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if config.DEVICE=='cuda' and not torch.cuda.is_available(): logging.warning("CUDA device selected but not available!")

    model_checkpoint = config.MODEL_PARAMS['model_checkpoint']
    learning_rate = config.MODEL_PARAMS['learning_rate']
    batch_size = config.MODEL_PARAMS['batch_size']
    num_epochs = config.MODEL_PARAMS['epochs'] # Use actual epochs now
    weight_decay = config.MODEL_PARAMS['weight_decay']
    gradient_accumulation_steps = config.MODEL_PARAMS['gradient_accumulation_steps']
    model_save_dir = config.PATHS['MODELS_DIR']
    os.makedirs(model_save_dir, exist_ok=True)

    logging.info(f"Loading feature extractor for: {model_checkpoint}")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        logging.info("Feature extractor loaded.")
    except Exception as e: logging.error(f"Could not load FE: {e}",exc_info=True); sys.exit(1)

    try:
        train_dataloader, val_dataloader, test_dataloader, train_size = setup_dataloaders(feature_extractor)
    except Exception as e: logging.error(f"Failed setup dataloaders: {e}",exc_info=True); sys.exit(1)

    logging.info(f"Loading model: {model_checkpoint}")
    try:
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        # Verify head replacement if needed (check attribute name)
        classifier_attr = 'classifier' # VERIFY
        if hasattr(model, classifier_attr):
             original_classifier = getattr(model, classifier_attr)
             if isinstance(original_classifier, nn.Linear):
                 in_features = original_classifier.in_features
                 setattr(model, classifier_attr, nn.Linear(in_features, num_labels))
                 logging.info(f"Replaced classifier head '{classifier_attr}'.")
             else: logging.warning(f"Head '{classifier_attr}' not nn.Linear.")
        else: logging.warning(f"Attribute '{classifier_attr}' not found.")
        model.to(device)
        logging.info("Model loaded and moved to device.")
    except Exception as e: logging.error(f"Failed load model: {e}",exc_info=True); sys.exit(1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Optional: Learning Rate Scheduler
    num_training_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
         optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    logging.info(f"Optimizer, Loss function, and LR Scheduler defined. Total training steps: {num_training_steps}")

    best_val_metric = float('inf'); metric_to_monitor = 'hamming_loss'
    start_time = time.time()
    logging.info(f"--- Starting Training for {num_epochs} epochs ---")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logging.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device, gradient_accumulation_steps, scheduler) # Pass scheduler
        eval_metrics = evaluate(model, val_dataloader, criterion, device)
        if not eval_metrics: logging.warning(f"Epoch {epoch+1}: Eval failed."); continue
        current_val_metric = eval_metrics.get(metric_to_monitor, float('inf'))
        if current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            save_path = os.path.join(model_save_dir, f"wav2vec2bert_finetuned_best.pth")
            try: torch.save(model.state_dict(), save_path); logging.info(f"Val metric improved ({metric_to_monitor}={current_val_metric:.4f}). Saved best model to {save_path}")
            except Exception as e: logging.error(f"Failed save checkpoint: {e}",exc_info=True)
        else: logging.info(f"Val metric did not improve ({metric_to_monitor}={current_val_metric:.4f}). Best: {best_val_metric:.4f}")
        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

    total_training_time = time.time() - start_time
    logging.info(f"--- Training Finished in {total_training_time / 60:.2f} minutes ---")

    logging.info("\n--- Evaluating on Test Set using Best Model ---")
    best_model_path = os.path.join(model_save_dir, f"wav2vec2bert_finetuned_best.pth")
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logging.info(f"Loaded best checkpoint from {best_model_path}")
            test_metrics = evaluate(model, test_dataloader, criterion, device)
            logging.info(f"\n--- Final Test Set Results ---")
            if test_metrics:
                 for metric_name, metric_value in test_metrics.items(): logging.info(f"Test {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
            else: logging.info("Test evaluation failed.")
        except Exception as e: logging.error(f"Failed load best model or evaluate test set: {e}",exc_info=True)
    else: logging.warning(f"Best checkpoint not found at {best_model_path}. Skipping test evaluation.")


# --- Make the script runnable ---
if __name__ == '__main__':
    run_training()