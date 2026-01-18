##########################################
# Train TTPD (Truth/Polarity Direction) Classifiers
##########################################

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import sys
from datetime import datetime
from config import JobConfig
from classifier_classes import TTPDClassifier

def add_ttpd_predictions_to_model_knowledge_files(classifier, layer_idx, model_name, train_idx, test_idx):
    """Add TTPD classifier predictions to model knowledge files."""
    # Load activations for the best layer
    acts = config.load_prepared_layer(model_name, layer_idx)
    
    # Get predictions for all data
    predictions = classifier.predict_proba(acts)
    prob_positive = predictions[:, 1]  # Extract probability of positive class only
    
    # Get prepared metadata to understand data structure
    metadata = config.load_prepared_metadata(model_name)
    dataset_info = metadata['dataset_info']  # List of (dataset_name, version, start_idx, end_idx)
    
    # Update each model knowledge file with the corresponding predictions
    for dataset_name, version, start_idx, end_idx in dataset_info:
        # Load the model knowledge file
        try:
            mk_df = config.load_model_knowledge_dataset(dataset_name, version)
        except FileNotFoundError:
            continue
        
        # Get predictions for this dataset's data
        dataset_predictions = prob_positive[start_idx:end_idx]
        
        # Add new column
        mk_df['prob_ttpd'] = dataset_predictions
        
        # Save updated model knowledge file
        mk_path = config.get_model_knowledge_dataset_path(dataset_name, version)
        mk_df.to_csv(mk_path, index=False)
        print(f"  Updated: {dataset_name}_{version:03d}.csv with TTPD predictions")

def train_ttpd_classifier():
    """Train TTPD classifiers."""
    model_name = config.model_name_clean
    
    # Load prepared metadata
    print("Loading prepared metadata")
    try:
        metadata = config.load_prepared_metadata(model_name)
        labels_raw = metadata['labels']
        polarities_raw = metadata['polarities']
        train_idx = metadata['train_idx']
        test_idx = metadata['test_idx']
        available_layers = metadata['available_layers']
        test_split_ratio = metadata['metadata']['test_split_ratio']
    except FileNotFoundError:
        print("Prepared data not found! Please run prepare_training_and_test_data.py first.")
        return
    
    # Convert labels and polarities to -1/1 format for TTPD classifier
    labels = np.array([1 if label == 1 else -1 for label in labels_raw])
    polarities = np.array([1 if pol == 1 else -1 for pol in polarities_raw])
    
    if len(labels) == 0:
        print("No training data found!")
        return
    
    best_accuracy = 0
    best_layer = None
    best_classifier = None
    layer_results = []
    
    # Try all available layers
    print("Training classifiers")
    for layer_idx in available_layers:
        print(f"Loading activations for layer {layer_idx}")
        try:
            acts = config.load_prepared_layer(model_name, layer_idx)
        except FileNotFoundError:
            print(f"Layer {layer_idx} data not found, skipping...")
            continue
        
        # Skip empty layers
        if acts.shape[0] == 0:
            continue
        
        try:
            # Create and train classifier
            classifier = TTPDClassifier(config.normalize_activations)
            classifier.layer = layer_idx
            
            # Train on training set
            train_acts = acts[train_idx]
            train_labels = labels[train_idx]
            train_polarities = polarities[train_idx]
            
            print(f"Training classifier for layer {layer_idx}")
            classifier.train(
                train_acts, 
                torch.tensor(train_labels, dtype=torch.long), 
                torch.tensor(train_polarities, dtype=torch.long)
            )
            
            # Evaluate
            print(f"Evaluating classifier for layer {layer_idx}")
            if test_idx:
                test_acts = acts[test_idx]
                test_labels = labels[test_idx]
                predictions = classifier.predict(test_acts)
                accuracy = accuracy_score(test_labels, predictions)
            else:
                # Use training accuracy if no test set
                predictions = classifier.predict(train_acts)
                accuracy = accuracy_score(train_labels, predictions)
            
            layer_results.append((layer_idx, accuracy))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_idx
                best_classifier = classifier
                
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as e:
            print(f"Skipping layer {layer_idx} due to numerical error: {e}")
            continue
    
    # Save best classifier and add predictions to model knowledge files
    if best_classifier:
        config.save_classifier(best_classifier, f"ttpd_best", model_name)
        print(f"Best TTPD classifier: Layer {best_layer}, Accuracy: {best_accuracy:.4f}")
        print(f"Saved to classifiers/ with date {config.job_start_date}")
        
        # Apply best classifier to all data and update model knowledge files
        print(f"Adding TTPD predictions to model knowledge files using layer {best_layer}")
        add_ttpd_predictions_to_model_knowledge_files(best_classifier, best_layer, model_name, train_idx, test_idx)

def main():
    """Main training function."""
    train_ttpd_classifier()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
