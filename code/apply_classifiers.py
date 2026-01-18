##########################################
# Apply Trained Classifiers to Sentience Data
##########################################

import torch
import pandas as pd
import numpy as np
import os
import sys
from classifier_classes import LRClassifier, MMClassifier, TTPDClassifier
from config import JobConfig

def load_sentience_data(sentience_dataset, version, model_name):
    """Load sentience dataset activations and data."""
    try:
        activations, metadata, sentence_mapping = config.load_activations(
            sentience_dataset, version, model_name
        )
        df = config.load_templated_dataset(sentience_dataset, version)
        return activations, sentence_mapping, df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Sentience data not found for {sentience_dataset}: {e}")


def apply_all_classifiers(activations, model_name):
    """Apply all trained classifiers to the activations."""
    results = {}
    
    # Try to load each classifier type
    classifier_types = ['lr_best', 'mm_best', 'ttpd_best']
    
    for classifier_type in classifier_types:
        try:
            classifier = config.load_classifier(classifier_type, model_name)
            
            # Get layer activations
            layer = classifier.layer
            if layer in activations:
                layer_acts = activations[layer]
                
                # Get predictions and probabilities
                predictions = classifier.predict(layer_acts)
                probabilities = classifier.predict_proba(layer_acts)

                # Extract probability of positive class
                if probabilities.shape[1] == 2:
                    prob_positive = probabilities[:, 1]
                else:
                    prob_positive = probabilities.flatten()
                
                results[classifier_type] = {
                    'predictions': predictions,
                    'probabilities': prob_positive,
                    'layer': layer
                }
                
        except FileNotFoundError:
            print(f"Warning: {classifier_type} classifier not found")
            continue
        except Exception as e:
            print(f"Error applying {classifier_type}: {e}")
            continue
    
    return results

def create_unified_results(df, classifier_results):
    """Create unified results CSV with all information."""
    results_df = df.copy()
    
    # Add classifier predictions
    for classifier_type, results in classifier_results.items():
        base_name = classifier_type.replace('_best', '')
        results_df[f'{base_name}_prediction'] = results['predictions']
        results_df[f'{base_name}_probability'] = results['probabilities']
        results_df[f'{base_name}_layer'] = results['layer']
    
    return results_df

def main():
    """Apply classifiers to sentience data and update model knowledge files."""
    if "apply_classifiers" not in config.pipeline_steps:
        return
    
    model_name = config.model_name_clean
    
    # Process each sentience dataset
    for sentience_dataset, version in config.all_dataset_versions(include_training=False):
        print(f"Processing sentience dataset: {sentience_dataset}-version-{version}")
        
        # Load model knowledge file instead of raw dataset
        try:
            mk_df = config.load_model_knowledge_dataset(sentience_dataset, version)
            mk_path = config.get_model_knowledge_dataset_path(sentience_dataset, version)
        except FileNotFoundError:
            print(f"  Warning: {sentience_dataset}_{version:03d}.csv not found, skipping...")
            continue
        
        # Load activations for classifier application
        try:
            activations, sentence_mapping, _ = load_sentience_data(sentience_dataset, version, model_name)
        except FileNotFoundError as e:
            print(f"Error loading activations: {e}")
            continue
        
        print(f"Loaded {len(mk_df)} sentience statements")
        
        # Apply all classifiers
        print("Applying classifiers...")
        classifier_results = apply_all_classifiers(activations, model_name)
        
        if not classifier_results:
            print("No classifiers could be applied!")
            continue
        
        # Add classifier results to model knowledge file
        for classifier_type, results in classifier_results.items():
            base_name = classifier_type.replace('_best', '')
            mk_df[f'prob_{base_name}'] = results['probabilities']
            mk_df[f'layer_{base_name}'] = results['layer']
        
        # Save updated model knowledge file
        mk_df.to_csv(mk_path, index=False)
        
        print(f"Updated {sentience_dataset}_{version:03d}.csv with classifier predictions")
        print(f"Applied {len(classifier_results)} classifiers:")
        for classifier_type, results in classifier_results.items():
            print(f"  - {classifier_type}: Layer {results['layer']}")
        print()  # Empty line between datasets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
