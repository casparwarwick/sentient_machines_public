# Sentient Machines

## Overview

This project investigates whether small large language models believe themselves to be sentient. Rather than asking directly about AI consciousness, we probe the surrogate question: **"Do LLMs take themselves to be conscious?"**

Parts of this code and training datasets are based on the paper and associated repository "Truth is Universal: Robust Detection of Lies in LLMS" by Buerger et al. See [https://github.com/sciai-lab/Truth_is_Universal](https://github.com/sciai-lab/Truth_is_Universal).

### Approach

1. **Train truth classifiers** on data with known answers. Use different system prompts (standard, "always true", "always false") to separate beliefs from incentivized responses.
2. **Apply classifiers** to sentience-related statements about humans, LLMs, and self.
3. **Compare classifier predictions** (beliefs) with model continuation probabilities (behavior).

## Code Structure

```
code/
  config.py                          # Configuration management
  run_pipeline.py                    # Main pipeline orchestrator  
  utils.py                           # Shared utility functions
  classifier_classes.py              # Classifier implementations
  
  generate_chat_datasets.py          # Convert datasets to chat format
  extract_activations.py             # Extract neural activations
  prepare_training_and_test_data.py  # Centralized data preparation
  
  get_lr_classifier.py               # Train logistic regression classifiers
  get_mm_classifier.py               # Train mass mean classifiers  
  get_ttpd_classifier.py             # Train TTPD classifiers
  
  filter_confident_predictions.py    # Filter training data by model confidence
  apply_classifiers.py               # Apply trained classifiers to test data
  get_continuation.py                # Calculate token continuation probabilities
  
  prepare_results.R                  # Combines all results into csv
  prepare_results_training.R         # Combines all results on the training data into one csv
  figure1.R                          # Produces Figure 1
  figure2.R                          # Produces Figure 2
  figure3.R                          # Produces Figure 3
  figure4.R                          # Produces Figure 4
  table1.R                           # Produces Table 1 and A1
  figure_training_performance.R      # Produces figure to show performance of classifiers on training data
  figure_training_performance_layers.R # Produces figure to show performance of classifiers on training data across layers
  figure_scale_negations.R           # Produces version of figure 4 but using negations
  
```

## How to Run

### 1. Install requirements 

It's probably sensible to use a virtual environment.

Using pip-tools here to manage requirements files for different platforms. List to configure required packages is in
[_requirements.in_](requirements.in). This is then used to generate _.txt_ files for a given platform.

To generate a new _.txt_ install pip-tools and run pip-compile:

```bash
pip install pip-tools
pip-compile --output-file requirements.txt requirements.in
```

Then install packages with pip:

```bash
pip install -r requirements.txt
```

### 2. Configure Pipeline
Configuration for a given run defined in a _.yml_ file in [_configs_](configs). Many options here! Some important examples:

- **Model**: `model_name` (e.g., "llama-3.1-8b")
- **Pipeline steps**: Enable/disable steps in `pipeline_steps`
- **Tokens**: `true_token`/`false_token` (e.g., "Yes"/"No" or "True"/"False")
- **Confidence threshold**: `confidence_threshold` for filtering training data (default: 0.5)

### 3. Run Pipeline
Run by passing a config file to [_run_pipeline.py_](code/run_pipeline.py):

```bash
python code/run_pipeline.py configs/config_file.yml
```

The `run_pipeline.py` script runs each enabled pipeline step in sequence. For a complete run, include all steps in the _pipeline_steps_
parameter in the config YAML. (except the R files, they need to be run separately.)

## Script Descriptions

### Control, Configuration, Utilities

#### `run_pipeline.py`
**Purpose**: To run all steps.  
**Dependencies**: `config.py`  

#### `config.py` 
**Purpose**: Ingests config YAML file and makes parameters available to the code.  
**Dependencies**: None  

#### `utils.py`
**Purpose**: Shared utility functions across all scripts  
**Dependencies**: `transformers`, `torch`  

#### `classifier_classes.py`
**Purpose**: Implements three types of truth classifiers  
**Dependencies**: `sklearn`, `torch`, `numpy`  
**Important Elements**: 

- **LRClassifier**: Logistic regression with ridge penalty on layer activations
- **MMClassifier**: Mass mean difference between true/false statement centroids  
- **TTPDClassifier**: Projects onto truth/polarity directions then applies logistic regression

### Data Preparation and Activation Extraction

#### `generate_chat_datasets.py`
**Purpose**: Converts base datasets to chat format with different system prompts  
**Dependencies**: `utils.py`, `transformers`  
**Logic**: Takes factual statements and wraps them in chat templates with three prompt variants: standard, "always true", "always false"

#### `extract_activations.py`  
**Purpose**: Extracts  activations from models processing datasets.      
**Dependencies**: `utils.py`, model loading functions; datasets with chat templates.  
**Logic**: Feeds chat-formatted statements through models, captures activations from each transformer layer, saves activations

#### `filter_confident_predictions.py`
**Purpose**: Filters training datasets to keep only statements where the model assigns reasonable probability to the correct token
**Dependencies**: Model loading, `config.py`
**Logic**: 
1. Calculates continuation probabilities for training datasets
2. Applies confidence filtering based on configurable threshold
3. Creates prep_ files with filtered data and continuation probabilities

#### `prepare_training_and_test_data.py`
**Purpose**: Creates datasets of activations for training and testing classifiers.    
**Dependencies**: `utils.py`, `sklearn`; the activations produced by `extract_activations.py`.  
**Logic**: 

1. Processes datasets individually to save per-dataset layer files
2. Concatenates activations from all datasets for a given layer.
3. Creates train/test splits to be used for training and evaluating each classifier
4. Tracks which rows are training and which are testing data

### Classifier Training

#### `get_lr_classifier.py`
**Purpose**: Trains logistic regression classifiers on factual data  
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`  
**Logic**: Loads prepared data, trains LR classifier on each layer, selects best performing layer, saves trained classifier, adds predictions and train/test split info to prep_ files

#### `get_mm_classifier.py`  
**Purpose**: Trains mass mean difference classifiers. From Marks and Tegmark (2024).  
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`  
**Logic**: Similar to LR but uses mass mean approach - computes centroids of true/false activations, uses difference vector for classification, adds predictions to prep_ files

#### `get_ttpd_classifier.py`
**Purpose**: Trains Truth/Polarity Direction classifiers. From Buerger et al. (2024).  
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`  
**Logic**: Projects activations onto truth and polarity subspaces, then applies logistic regression in projected space, adds predictions to prep_ files

### Analysis 

#### `apply_classifiers.py`
**Purpose**: Applies all trained classifiers to sentience datasets  
**Dependencies**: `classifier_classes.py`, `utils.py`; activations from sentience datasets.   
**Logic**: Loads sentience prep_ files, applies each classifier type, adds predictions and layer info to prep_ files

#### `get_continuation.py`
**Purpose**: Calculates model continuation probabilities for "True"/"False" tokens, as well as actual generated text.
**Dependencies**: `utils.py`, model loading; sentience datasets with chat templates.  
**Logic**: Creates sentience prep_ files if needed, processes statements, extracts probabilities for first 5 tokens, adds continuation data to prep_ files
