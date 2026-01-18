##########################################
# Filter Confident Predictions
##########################################

# Filter datasets to keep only statements where the model assigns reasonable probability 
# to the correct token (True/False), ensuring classifiers train only on statements
# the model "knows".

import traceback

import torch
import pandas as pd
import os
import sys
import math
import torch.nn.functional as F
from config import JobConfig

#=================================
# 1. Define functions
#=================================

def get_token_probabilities_for_batch(batch, model, tokenizer, device, true_token_id, false_token_id):
    """Get True/False token probabilities for a batch of statements using sum of first 5 generated tokens."""
    if config.pad_all_inputs:
        padding = "max_length"
        max_length = config.get_padding_length()
    else:
        padding = "longest"
        max_length = None
    
    # Tokenize input with attention mask
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=padding, max_length=max_length,
                      return_attention_mask=True).to(device)
    
    batch_size = inputs['input_ids'].shape[0]
    
    # Initialize probability arrays for 5 tokens
    probs_true_all_tokens = []
    probs_false_all_tokens = []
    
    # Generate and analyze 5 tokens sequentially
    current_inputs = inputs
    
    for token_position in range(5):
        with torch.no_grad():
            outputs = model(**current_inputs)
        
        # Get logits for the last token position
        logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Extract probabilities for True and False tokens at this position
        probs_true_token = probs[:, true_token_id].cpu().float().numpy()
        probs_false_token = probs[:, false_token_id].cpu().float().numpy()
        
        probs_true_all_tokens.append(probs_true_token)
        probs_false_all_tokens.append(probs_false_token)
        
        # For next iteration, sample the next token and append to input
        # Use the most likely token (greedy sampling) to match get_continuation.py behavior
        next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Append next token to input_ids
        current_inputs['input_ids'] = torch.cat([current_inputs['input_ids'], next_token_ids], dim=-1)
        
        # Extend attention mask
        current_inputs['attention_mask'] = torch.cat([
            current_inputs['attention_mask'], 
            torch.ones(batch_size, 1, device=device)
        ], dim=-1)
    
    # Sum probabilities across all 5 tokens for each sample in the batch
    probs_true_sum = sum(probs_true_all_tokens[i] for i in range(5))
    probs_false_sum = sum(probs_false_all_tokens[i] for i in range(5))
    
    return probs_true_sum, probs_false_sum

def filter_dataset_by_confidence(dataset_name, model, tokenizer, device, true_token_id, false_token_id):
    """Filter a single _001 dataset to keep only confident predictions."""
    print(f"Processing dataset: {dataset_name}")
    
    # Load _001 version (standard prompts)
    try:
        df_001 = config.load_templated_dataset(dataset_name, 1)
    except FileNotFoundError:
        print(f"  Warning: Processed dataset {dataset_name}_001 not found, skipping...")
        return 0, 0
    
    # Initialize probability columns
    df_001['prob_true'] = 0.0
    df_001['prob_false'] = 0.0
    
    total_rows = len(df_001)
    n_batches = math.ceil(total_rows / config.batch_size)
    start_idx = 0
    
    # Process each batch to get probabilities
    for batch_num in range(n_batches):
        batch = df_001.iloc[start_idx:min(len(df_001), start_idx + config.batch_size)]
        start_idx += config.batch_size
        
        statements = list(batch['statement'])
        
        probs_true, probs_false = get_token_probabilities_for_batch(
            statements, model, tokenizer, device, true_token_id, false_token_id
        )
        
        # Store probabilities (now representing sum of first 5 tokens)
        df_001.loc[batch.index, 'prob_true'] = probs_true
        df_001.loc[batch.index, 'prob_false'] = probs_false
        
        print(f"  [{start_idx}/{total_rows}] Processed batch {batch_num}...")
    
    # Confidence filtering criteria: require minimum probability and correct direction
    confident_mask = (
        # For true labels: prob_true > threshold AND prob_true > prob_false
        ((df_001['label'] == 1) & (df_001['prob_true'] > config.confidence_threshold) & (df_001['prob_true'] > df_001['prob_false'])) |
        # For false labels: prob_false > threshold AND prob_false > prob_true  
        ((df_001['label'] == 0) & (df_001['prob_false'] > config.confidence_threshold) & (df_001['prob_false'] > df_001['prob_true']))
    )
    
    confident_indices = df_001[confident_mask].index.tolist()
    
    print(f"  Kept {len(confident_indices)}/{len(df_001)} confident predictions ({len(confident_indices)/len(df_001)*100:.1f}%)")
    
    # Load and filter all three versions using the confident indices
    for version in [1, 2, 3]:
        df_version = config.load_templated_dataset(dataset_name, version)
        df_filtered = df_version.loc[confident_indices].copy()
        
        # Add continuation probabilities only for version 1 (_001)
        if version == 1:
            df_filtered['prob_true'] = df_001.loc[confident_indices, 'prob_true']
            df_filtered['prob_false'] = df_001.loc[confident_indices, 'prob_false']
            print(f"  Added continuation probabilities for version {version}")
        
        # Save filtered dataset with prep_ prefix
        prep_path = config.get_model_knowledge_dataset_path(dataset_name, version)
        df_filtered.to_csv(prep_path, index=False)
        if version == 1:
            print(f"  Saved: prep_{dataset_name}_{version:03d}.csv ({len(df_filtered)} rows, includes prob_true/prob_false)")
        else:
            print(f"  Saved: prep_{dataset_name}_{version:03d}.csv ({len(df_filtered)} rows)")
    
    return len(confident_indices), len(df_001)

def main():
    """Main function to filter all training datasets by confidence."""
    if "filter_confident_predictions" not in config.pipeline_steps:
        return
    
    print("Starting confidence-based dataset filtering...")
    print(f"Model: {config.model_name}")
    print(f"True token: '{config.true_token}', False token: '{config.false_token}'")
    
    # Get token IDs for True and False
    true_token_id, false_token_id = config.get_true_false_token_ids()
    
    # Load model
    model, tokenizer, device = config.load_model_and_tokenizer_standardized()
    
    print(f"True token ID: {true_token_id}, False token ID: {false_token_id}")
    print("Filtering criteria (using sum of first 5 tokens):")
    print(f"  - For label=1: Sum_P({config.true_token}) > {config.confidence_threshold} AND Sum_P({config.true_token}) > Sum_P({config.false_token})")
    print(f"  - For label=0: Sum_P({config.false_token}) > {config.confidence_threshold} AND Sum_P({config.false_token}) > Sum_P({config.true_token})")
    
    # Create prep datasets directory
    os.makedirs(config.model_knowledge_datasets_dir, exist_ok=True)
    
    # Process each training dataset
    total_kept = 0
    total_original = 0
    first_dataset_processed = False
    
    for dataset_name in config.base_training_datasets:
        try:
            kept, original = filter_dataset_by_confidence(
                dataset_name, model, tokenizer, device, true_token_id, false_token_id
            )
            total_kept += kept
            total_original += original
            print()  # Empty line between datasets
            
            # Early exit if first dataset has zero confident predictions
            if not first_dataset_processed:
                first_dataset_processed = True
                if kept == 0 and original > 0:
                    print(f"ERROR: First dataset '{dataset_name}' produced 0 confident predictions!")
                    print("This indicates a configuration issue (likely generation parameters).")
                    print("Check your true_token, false_token, temperature, and other generation settings.")
                    print("Exiting early to avoid wasting compute time.")
                    sys.exit(1)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            print(traceback.format_exc())
            continue
    
    if total_original > 0:
        print(f"Overall filtering results:")
        print(f"Kept {total_kept}/{total_original} confident predictions ({total_kept/total_original*100:.1f}%)")
    else:
        print("No datasets were processed successfully.")

#=================================
# 2. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
