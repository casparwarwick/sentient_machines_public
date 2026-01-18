##########################################
# Extract Neural Activations Per Dataset
##########################################
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
from config import JobConfig

# Configuration
LAYERS_TO_EXTRACT = list(range(1, 81)) # 81 because python is weird at counting
MAX_LENGTH = 512

def extract_activations_for_batch(model, tokenizer, batch, device, layers_to_extract, padding_length):
    """Extract activations for a batch of statements."""
    if config.pad_all_inputs:
        padding = "max_length"
    else:
        padding = "longest"

    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=padding, max_length=padding_length,
                      return_attention_mask=True).to(device)
    activations = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states has size batch_size x sequence_length x layer_output_size
            last_token_activations = hidden_states[:, -1, :].detach().cpu().to(torch.float16)
            activations[layer_idx] = last_token_activations
        return hook
    
    hooks = []
    for layer_idx in layers_to_extract:
        if layer_idx <= len(model.model.layers):
            layer = model.model.layers[layer_idx - 1]
            hook = layer.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)
    
    with torch.no_grad():
        model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def process_dataset(dataset_name, dataset_version, model_name, model, tokenizer, device, padding_length, use_model_knowledge=False):
    """Process a single dataset and save its activations."""
    if "extract_activations" not in config.pipeline_steps:
        return

    # Check if activations already exist
    activation_dir = config.get_activations_dir(dataset_name, dataset_version, model_name)
    if os.path.exists(activation_dir):
        print(f"Activations for {dataset_name}-version-{dataset_version} with {model_name} already exist at {activation_dir}, skipping...")
        return
    
    # Load dataset (use model knowledge version for training datasets if available)
    try:
        if use_model_knowledge:
            try:
                df = config.load_model_knowledge_dataset(dataset_name, dataset_version)
                print(f"Using model knowledge dataset.")
            except FileNotFoundError:
                print(f"Model knowledge dataset not found, falling back to templated dataset")
                df = config.load_templated_dataset(dataset_name, dataset_version)
        else:
            df = config.load_templated_dataset(dataset_name, dataset_version)
    except FileNotFoundError:
        print(f"Warning: Dataset {dataset_name} not found, skipping...")
        return
    
    # Model, tokenizer, and device are now passed as parameters
    # Prepare data structures
    all_activations = {layer: [] for layer in LAYERS_TO_EXTRACT}
    sentence_mapping = []
    
    # Process each batch
    n_batches = math.ceil(len(df) / config.batch_size)
    start_idx = 0

    for batch_num in range(n_batches):
        batch = df.iloc[start_idx:min(len(df), start_idx + config.batch_size)]
        start_idx += config.batch_size

        statements = list(batch['statement'])

        # Extract activations
        activations = extract_activations_for_batch(
            model, tokenizer, statements, device, LAYERS_TO_EXTRACT, padding_length
        )

        # Store activations
        for layer_idx in LAYERS_TO_EXTRACT:
            if layer_idx in activations:
                all_activations[layer_idx].append(activations[layer_idx])

        # Store mapping
        sentence_mapping += [{
            'sentence_idx': idx,
            'statement': row['statement'],
            'label': row.get('label', None),
            'polarity': row.get('polarity', None),
            'dataset_id': row.get('dataset_id', dataset_name),
            'sentence_id': row.get('sentence_id', idx)
        } for idx, row in batch.iterrows()]
    
    # Convert to tensors
    for layer_idx in LAYERS_TO_EXTRACT:
        if all_activations[layer_idx]:
            all_activations[layer_idx] = torch.cat(all_activations[layer_idx])
        else:
            all_activations[layer_idx] = torch.empty(0)
    
    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'dataset_version': dataset_version,
        'layers_extracted': LAYERS_TO_EXTRACT,
        'num_statements': len(df),
        'max_length': padding_length,
        'device': str(device),
        'date': config.job_start_date
    }
    
    # Save activations
    save_dir = config.save_activations(
        all_activations, metadata, sentence_mapping, 
        dataset_name, dataset_version, model_name
    )
    
    print(f"Saved activations for {dataset_name} to {save_dir}")

def main():
    """Main extraction function."""
    model_name = config.model_name
    
    # Load model once for all datasets
    print(f"Loading model: {model_name}")
    model, tokenizer, device = config.load_model_and_tokenizer_standardized()
        
    if config.pad_all_inputs:
        max_input_length = config.get_padding_length()
        print(f"Padding all inputs to length {max_input_length}")
    else:
        max_input_length = MAX_LENGTH
    
    try:
        # Process each training dataset separately (use model knowledge datasets if available)
        for dataset_name, version in config.all_dataset_versions(include_sentience=False):
            print(f"Processing dataset: {dataset_name}-version-{version}")
            process_dataset(dataset_name, version, model_name, model, tokenizer, device, max_input_length, use_model_knowledge=True)
        
        # Process sentience dataset(s) separately (always use templated datasets for testing)
        for dataset_name, version in config.all_dataset_versions(include_training=False):
            print(f"Processing sentience dataset: {dataset_name}-version-{version}")
            process_dataset(dataset_name, version, model_name, model, tokenizer, device, max_input_length, use_model_knowledge=False)
        
        print("Activation extraction completed!")
        
    finally:
        # Clean up GPU memory at the end
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
