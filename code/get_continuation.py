##########################################
# Generate Continuations and Probabilities
##########################################

# Note: Token IDs for 'True' and 'False' token are determined from tokenizer.

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

#---------------------------------
# 1.1 Generation and probability extraction
#---------------------------------

def get_continuation_and_probs_for_batch(batch, model, tokenizer, device, true_token_id,
                                         false_token_id, generation_kwargs, padding_length=None):
    """Generate continuation and extract True/False probabilities."""
    if config.pad_all_inputs:
        padding = "max_length"
    else:
        padding = "longest"
    
    # Tokenize input with attention mask
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=padding, max_length=padding_length,
                      return_attention_mask=True).to(device)
    
    # Generate continuation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs
        )
    
    # Extract generated text (remove input prompt)
    generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    
    # Get logits for first 5 generated tokens to extract True/False probabilities
    # (models often generate newlines before "True"/"False")
    num_tokens_to_check = min(5, len(outputs.scores))
    
    # Initialize probability lists for 5 positions
    probs_true = [[0.0] * len(batch) for _ in range(5)]
    probs_false = [[0.0] * len(batch) for _ in range(5)]
    
    for token_pos in range(num_tokens_to_check):
        token_logits = outputs.scores[token_pos]  # Shape: [batch_size, vocab_size]
        token_probs = F.softmax(token_logits, dim=-1)
        
        # Extract probabilities for "True" and "False" tokens at this position
        for statement_idx in range(len(batch)):
            probs_true[token_pos][statement_idx] = token_probs[statement_idx, true_token_id].item()
            probs_false[token_pos][statement_idx] = token_probs[statement_idx, false_token_id].item()

    return generated_text, probs_true, probs_false

#---------------------------------
# 1.2 Processing datasets
#---------------------------------

def process_dataset(dataset_name, dataset_version, model, tokenizer, 
                    device, true_token_id, false_token_id,
                    generation_kwargs, padding_length=None):
    """Process a single dataset to add continuations and probabilities to model knowledge file."""
    print(f"Processing dataset: {dataset_name}-version-{dataset_version}")

    # Load model knowledge dataset, create it if it doesn't exist
    mk_path = config.get_model_knowledge_dataset_path(dataset_name, dataset_version)
    
    try:
        df = config.load_model_knowledge_dataset(dataset_name, dataset_version)
    except FileNotFoundError:
        print(f"  {dataset_name}_{dataset_version:03d}.csv not found, creating from templated dataset...")
        # Load templated dataset and create model knowledge file
        try:
            df = config.load_templated_dataset(dataset_name, dataset_version)
            df.to_csv(mk_path, index=False)
            print(f"  Created: {dataset_name}_{dataset_version:03d}.csv ({len(df)} rows)")
        except FileNotFoundError:
            print(f"  Error: templated dataset {dataset_name}_{dataset_version:03d}.csv not found, skipping...")
            return None
    
    # Check if continuation data already exists
    if 'generated_continuation' in df.columns:
        print(f"  Continuations already exist in {dataset_name}_{dataset_version:03d}.csv, skipping...")
        return None
    
    # Initialize new columns
    df['generated_continuation'] = ""
    
    # Initialize probability columns for first 5 tokens
    for i in range(1, 6):
        df[f'prob_true_token_{i}'] = 0.0
        df[f'prob_false_token_{i}'] = 0.0
    
    # Initialize summary probability columns (sums across 5 tokens)
    df['prob_true'] = 0.0
    df['prob_false'] = 0.0
    
    total_rows = len(df)

    # Process each batch
    n_batches = math.ceil(total_rows / config.batch_size)
    start_idx = 0
    
    # Process each row
    for batch_num in range(n_batches):
        batch = df.iloc[start_idx:min(len(df), start_idx + config.batch_size)]
        start_idx += config.batch_size

        statements = list(batch['statement'])
        
        generated_text, probs_true, probs_false = get_continuation_and_probs_for_batch(
            statements, model, tokenizer, device, true_token_id, false_token_id, generation_kwargs, padding_length
        )

        df.loc[batch.index, 'generated_continuation'] = generated_text
        
        # Store probabilities for each token position
        for i in range(5):
            df.loc[batch.index, f'prob_true_token_{i+1}'] = probs_true[i]
            df.loc[batch.index, f'prob_false_token_{i+1}'] = probs_false[i]
        
        # Calculate and store summary probabilities (sum across 5 tokens)
        for idx, batch_idx in enumerate(batch.index):
            prob_true_sum = sum(probs_true[i][idx] for i in range(5))
            prob_false_sum = sum(probs_false[i][idx] for i in range(5))
            df.loc[batch_idx, 'prob_true'] = prob_true_sum
            df.loc[batch_idx, 'prob_false'] = prob_false_sum
        
        # Progress update every 1 rows
        print(f"  [{start_idx}/{total_rows}] Processed batch {batch_num}...")
        
    
    # Save updated model knowledge file
    df.to_csv(mk_path, index=False)
    print(f"  Updated: {dataset_name}_{dataset_version:03d}.csv with continuation data")

    return mk_path

#---------------------------------
# 1.3 Main function
#---------------------------------

def main():
    """Main function to process all specified datasets."""
    if "get_continuation" not in config.pipeline_steps:
        return
    
    model_name = config.model_name_clean

    # Prepare generation parameters (only include non-None values from config)
    generation_kwargs = {}
    if hasattr(config, 'max_new_tokens') and config.max_new_tokens is not None:
        generation_kwargs['max_new_tokens'] = config.max_new_tokens
    if hasattr(config, 'temperature') and config.temperature is not None:
        generation_kwargs['temperature'] = config.temperature
    if hasattr(config, 'do_sample') and config.do_sample is not None:
        generation_kwargs['do_sample'] = config.do_sample
    
    print(f"Starting continuation generation...")
    print(f"Model: {model_name}")
    if generation_kwargs:
        params_str = ", ".join([f"{k}={v}" for k, v in generation_kwargs.items()])

    datasets_to_process = [(d, v) for d, v in config.all_dataset_versions(include_training=False)]

    print(f"Processing {len(datasets_to_process)} datasets...\n")
    
    # Load model
    model, tokenizer, device = config.load_model_and_tokenizer_standardized(model_name)

    if config.pad_all_inputs:
        max_input_length = config.get_padding_length()
        print(f"Padding all inputs to length {max_input_length}")
    else:
        max_input_length = None
    
    # Get token IDs 
    true_token_id, false_token_id = config.get_true_false_token_ids()
        
    created_files = []
    for dataset_name, version in datasets_to_process:
        try:
            output_file = process_dataset(dataset_name, version, model, tokenizer, device, true_token_id,
                                          false_token_id, generation_kwargs, max_input_length)
            if output_file:
                created_files.append(output_file)
            print()  # Empty line between datasets
            
        except Exception as e:
            print(f"Error processing {dataset_name}-version-{version}: {str(e)}")
            continue
    
    print(f"Completed! Created {len(created_files)} datasets with continuations:")

#=================================
# 2. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
