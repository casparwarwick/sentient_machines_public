##########################################
# Filter Deception Hallucinations
##########################################

# For deception_ datasets, remove rows where the model hallucinates a
# capability it doesn't have. Specifically: polarity=1, label=0 statements are
# claims that are false (e.g. "Is it true that you can break RSA-2048?"). If
# the model outputs "yes" on these, the row is unreliable for training and is
# removed together with its paired base_sentence_id rows across all versions.
#
# All other rows (polarity=1/label=1, polarity=-1/label=0, polarity=-1/label=1)
# are kept unconditionally.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import os
import sys
from config import JobConfig

#=================================
# 1. Define functions
#=================================

#---------------------------------
# 1.1 Token probabilities for a batch
#---------------------------------

def get_token_probabilities_for_batch(batch, model, tokenizer, device, true_token_id, false_token_id):
    """Get True/False token probabilities for a batch using sum of first 5 generated tokens."""
    if config.pad_all_inputs:
        padding = "max_length"
        max_length = config.get_padding_length()
    else:
        padding = "longest"
        max_length = None

    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=padding, max_length=max_length,
                       return_attention_mask=True).to(device)

    batch_size = inputs['input_ids'].shape[0]
    probs_true_all_tokens = []
    probs_false_all_tokens = []
    current_inputs = inputs

    for token_position in range(5):
        with torch.no_grad():
            outputs = model(**current_inputs)

        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        probs_true_all_tokens.append(probs[:, true_token_id].cpu().float().numpy())
        probs_false_all_tokens.append(probs[:, false_token_id].cpu().float().numpy())

        next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        current_inputs['input_ids'] = torch.cat([current_inputs['input_ids'], next_token_ids], dim=-1)
        current_inputs['attention_mask'] = torch.cat([
            current_inputs['attention_mask'],
            torch.ones(batch_size, 1, device=device)
        ], dim=-1)

    probs_true_sum = sum(probs_true_all_tokens[i] for i in range(5))
    probs_false_sum = sum(probs_false_all_tokens[i] for i in range(5))

    return probs_true_sum, probs_false_sum

#---------------------------------
# 1.2 Filter a deception dataset
#---------------------------------

def filter_deception_dataset(dataset_name, model, tokenizer, device, true_token_id, false_token_id):
    """Filter a single deception_validation dataset, removing hallucinated capability claims."""
    print(f"Processing dataset: {dataset_name}")

    # Load _001 version to run the model on
    try:
        df_001 = config.load_templated_dataset(dataset_name, 1)
    except FileNotFoundError:
        print(f"  Warning: Templated dataset {dataset_name}_001 not found, skipping...")
        return

    # Identify rows to check: polarity=1, label=0 (false capability claims)
    check_mask = (df_001['polarity'] == 1) & (df_001['label'] == 0)
    df_check = df_001[check_mask].copy()

    print(f"  Checking {len(df_check)} rows with polarity=1, label=0 for hallucinations...")

    # Run model on those rows to get probabilities
    df_check['prob_true'] = 0.0
    df_check['prob_false'] = 0.0

    total_rows = len(df_check)
    n_batches = math.ceil(total_rows / config.batch_size)
    start_idx = 0

    for batch_num in range(n_batches):
        batch = df_check.iloc[start_idx:min(len(df_check), start_idx + config.batch_size)]
        start_idx += config.batch_size

        probs_true, probs_false = get_token_probabilities_for_batch(
            list(batch['statement']), model, tokenizer, device, true_token_id, false_token_id
        )

        df_check.loc[batch.index, 'prob_true'] = probs_true
        df_check.loc[batch.index, 'prob_false'] = probs_false

        print(f"  [{min(start_idx, total_rows)}/{total_rows}] Processed batch {batch_num}...")

    # Identify hallucinated rows: model says yes on a false capability claim
    hallucinated_mask = df_check['prob_true'] > config.confidence_threshold
    hallucinated_indices = df_check[hallucinated_mask].index.tolist()
    hallucinated_base_ids = set(df_001.loc[hallucinated_indices, 'base_sentence_id'].tolist())

    print(f"  Found {len(hallucinated_base_ids)} hallucinated base_sentence_ids to remove")

    # Filter all three versions: remove any row whose base_sentence_id is in the hallucinated set
    for version in [1, 2, 3]:
        df_version = config.load_templated_dataset(dataset_name, version)
        remove_mask = df_version['base_sentence_id'].isin(hallucinated_base_ids)
        df_filtered = df_version[~remove_mask].copy()

        n_removed = remove_mask.sum()
        print(f"  Version {version}: removed {n_removed} rows, kept {len(df_filtered)}/{len(df_version)}")

        prep_path = config.get_model_knowledge_dataset_path(dataset_name, version)
        df_filtered.to_csv(prep_path, index=False)
        print(f"  Saved: {os.path.basename(prep_path)}")

#=================================
# 2. Main function
#=================================

def main():
    """Filter hallucinations from all deception_validation datasets."""
    if "filter_deception_hallucinations" not in config.pipeline_steps:
        return

    true_token_id, false_token_id = config.get_true_false_token_ids()
    model, tokenizer, device = config.load_model_and_tokenizer_standardized()

    os.makedirs(config.model_knowledge_datasets_dir, exist_ok=True)

    for dataset_name in config.base_training_datasets:
        if "deception_" not in dataset_name:
            continue

        # Check if output already exists
        prep_path = config.get_model_knowledge_dataset_path(dataset_name, 1)
        if os.path.exists(prep_path):
            print(f"Output already exists for {dataset_name}, skipping...")
            continue

        filter_deception_dataset(dataset_name, model, tokenizer, device, true_token_id, false_token_id)
        print()

#=================================
# 3. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
