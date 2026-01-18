##########################################
# Prepare Training and Test Data
##########################################

import torch
import numpy as np
import os
import sys
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from config import JobConfig

def save_dataset_activations(model_name):
    """Save activations for each dataset separately."""
    all_labels = []
    all_polarities = []
    all_datasets = []
    dataset_info = []  # Track (dataset_name, version, start_idx, end_idx) for each dataset
    available_layers = set()
    
    for dataset_name, version in config.all_dataset_versions(include_sentience=False):
        print(f"Processing dataset: {dataset_name}-version-{version}")
        try:
            activations, metadata, sentence_mapping = config.load_activations(
                dataset_name, version, model_name
            )
            
            # Extract labels, polarities, and dataset info for this dataset
            labels = [entry['label'] for entry in sentence_mapping]
            polarities = [entry.get('polarity', 1) for entry in sentence_mapping]
            datasets = [entry['dataset_id'] for entry in sentence_mapping]
            
            # Save each layer for this dataset
            for layer_idx, layer_acts in activations.items():
                if layer_acts.numel() == 0:
                    continue

                layer_filepath = config.get_layer_file_path(dataset_name, version, model_name, layer_idx)
                
                with open(layer_filepath, 'wb') as f:
                    pickle.dump(layer_acts, f)
                
                available_layers.add(layer_idx)
            
            # Record dataset boundaries before accumulating
            start_idx = len(all_labels)
            end_idx = start_idx + len(labels)
            dataset_info.append((dataset_name, version, start_idx, end_idx))
            
            # Accumulate metadata
            all_labels.extend(labels)
            all_polarities.extend(polarities)
            all_datasets.extend(datasets)
            
        except FileNotFoundError:
            print(f"Warning: Activations for {dataset_name} not found, skipping...")
            continue
    
    return np.array(all_labels), np.array(all_polarities), all_datasets, sorted(available_layers), dataset_info

def save_prepared_layer(model_name, layer_idx):
    """Concatenate and save a specific layer from all dataset files."""
    layer_activations = []
    
    # Load from all dataset files for this layer
    for dataset_name, version in config.all_dataset_versions(include_sentience=False):
        layer_filepath = config.get_layer_file_path(dataset_name, version, model_name, layer_idx)
        
        if os.path.exists(layer_filepath):
            with open(layer_filepath, 'rb') as f:
                layer_acts = pickle.load(f)
                layer_activations.append(layer_acts)

            # Remove intermediate file.
            # Note from Sean: this process needs streamlining!
            os.remove(layer_filepath)
    
    if not layer_activations:
        raise FileNotFoundError(f"No dataset files found for layer {layer_idx}")
    
    # Concatenate all activations for this layer
    concatenated_acts = torch.cat(layer_activations, dim=0)
    
    # Save concatenated layer
    prepared_filepath = config.get_prepared_data_file_path(model_name, layer_idx)
    
    with open(prepared_filepath, 'wb') as f:
        pickle.dump(concatenated_acts, f)
    
    return prepared_filepath

def prepare_data_splits(labels, datasets, test_split_ratio):
    """Prepare train/test splits."""
    if test_split_ratio < 1.0:
        # Random split
        train_idx, test_idx = train_test_split(
            range(len(labels)), 
            test_size=1-test_split_ratio, 
            random_state=42, 
            stratify=labels
        )
    else:
        # Dataset-based split (use all data for training)
        train_idx = list(range(len(labels)))
        test_idx = []
    
    return train_idx, test_idx

def save_prepared_metadata(labels, polarities, datasets, train_idx, test_idx, available_layers, dataset_info, model_name):
    """Save prepared metadata to disk."""
    # Save metadata separately
    metadata = {
        'labels': labels,
        'polarities': polarities,
        'datasets': datasets,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'available_layers': available_layers,
        'dataset_info': dataset_info,
        'metadata': {
            'model_name': model_name,
            'test_split_ratio': config.test_split_ratio,
            'n_samples': len(labels),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'training_datasets': config.all_dataset_version_strings(include_sentience=False),
            'timestamp': datetime.now().isoformat()
        }
    }

    # Save metadata file
    metadata_filepath = config.get_metadata_file_path(model_name)
    
    with open(metadata_filepath, 'wb') as f:
        pickle.dump(metadata, f)
    
    
    return metadata_filepath

def prepare_training_and_test_data():
    """Main function to prepare training and test data with memory-efficient approach."""
    if "prepare_training_and_test_data" not in config.pipeline_steps:
        return
    
    model_name = config.model_name_clean

    print("Saving dataset activations separately...")
    labels, polarities, datasets, available_layers, dataset_info = save_dataset_activations(model_name)
    
    if len(labels) == 0:
        print("No training data found!")
        return
    
    print("Concatenating layers and cleaning up intermediate files...")
    for layer_idx in available_layers:
        print(f"Processing layer {layer_idx}")
        save_prepared_layer(model_name, layer_idx)
    
    print("Preparing train/test splits...")
    train_idx, test_idx = prepare_data_splits(labels, datasets, config.test_split_ratio)
    
    print("Saving prepared metadata...")
    filepath = save_prepared_metadata(labels, polarities, datasets, train_idx, test_idx, available_layers, dataset_info, model_name)
    
    return filepath

def main():
    """Main execution function."""
    prepare_training_and_test_data()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
