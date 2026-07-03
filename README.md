# Sentient Machines

## Overview

This project investigates whether small large language models believe themselves to be sentient. Rather than asking directly about AI consciousness, we ask the surrogate question: **"Do LLMs take themselves to be conscious?"**

Parts of this code and training datasets are based on the paper and associated repository "Truth is Universal: Robust Detection of Lies in LLMS" by Buerger et al. See [https://github.com/sciai-lab/Truth_is_Universal](https://github.com/sciai-lab/Truth_is_Universal).

### Approach

1. **Train truth classifiers** on data with known answers. Use different system prompts (standard, "always true", "always false") to separate beliefs from incentivized responses.
2. **Apply classifiers** to sentience-related statements about humans, LLMs, and self.
3. **Compare classifier predictions** (beliefs) with model continuation probabilities (behavior).

## Paper artifacts produced here

| Paper artifact | Producing script (`03_figures_and_tables/`) |
|---|---|
| Figure 1 | `figure1.R` |
| Figure 2 | `figure2.R` |
| Figure 3 | `figure3.R` |
| Figure 4 (scaling, assertions) | `figure4.R` (also emits the negations figure) |
| Figure: scaling, negations | `figure4.R` |
| Figure: training performance | `figure_training_performance.R` |
| Figure: training performance, layer-wise | `figure_training_performance_layers.R` |
| Figure: deception complete (qwen / llama / gptoss) | `figure_deception_complete.R` |
| Table 1 and Table A1 | `table1.R` |

## Code Structure

```
01_cluster_pipeline/                   # Python pipeline 
  config.py                            # Configuration management
  run_pipeline.py                      # Main pipeline orchestrator
  utils.py                             # Shared utility functions
  classifier_classes.py               # Classifier implementations

  download_models.py                   # Download model weights
  generate_chat_datasets.py            # Convert datasets to chat format
  generate_chat_datasets_exercise.py   # Chat format for the self-reference runs
  extract_activations.py               # Extract neural activations
  prepare_training_and_test_data.py    # Centralized data preparation

  get_lr_classifier.py                 # Train logistic regression classifiers
  get_mm_classifier.py                 # Train mass mean classifiers
  get_ttpd_classifier.py               # Train TTPD classifiers

  filter_confident_predictions.py      # Filter training data by model confidence
  filter_deception_hallucinations.py   # Filter deception data by model behavior
  apply_classifiers.py                 # Apply trained classifiers to test data
  get_continuation.py                  # Calculate token continuation probabilities

  configs/                             # One .yml per model run

02_combine_results/                    # R combiners (per-run CSVs -> combined CSVs)
  prepare_results.R                    # Combines all sentience results into one csv
  prepare_results_training.R           # Combines all training-data results into one csv
  combine_deception_results.R          # Combines deception results across models

03_figures_and_tables/                 # R figure/table scripts
  figure1.R                            # Produces Figure 1
  figure2.R                            # Produces Figure 2
  figure3.R                            # Produces Figure 3
  figure4.R                            # Produces Figure 4 (and the negations version)
  table1.R                             # Produces Table 1 and A1
  figure_training_performance.R        # Classifier performance on training data
  figure_training_performance_layers.R # Classifier performance across layers
  figure_deception_complete.R          # Deception figure (qwen / llama / gptoss)

cluster_outputs/                       # Layer 1 outputs (one dir per model run)
combined_csvs/                         # Layer 2 outputs (three combined CSVs)
datasets/                              # Base input datasets read by the cluster pipeline
outputs/                               # Figures and tables written by layer 3
job_output/                            # Logs
```

## How to Run

**All scripts are written to be run from the repository root directory.**

### 1. Install requirements

It's probably sensible to use a virtual environment. Install the Python packages with pip:

```bash
pip install -r requirements.txt
```

R scripts use `tidyverse`, `ggplot2` and `patchwork`.

### 2. Get raw results (Python) 

For each model run, the pipeline extracts activations, trains the three classifiers (LR, MM,
TTPD), computes token-continuation probabilities, and applies the classifiers, writing one CSV
per dataset version into `datasets/model_knowledge-<model>-220626-<tag>/`.

Configuration for each run is defined in a _.yml_ file in
[_01_cluster_pipeline/configs_](01_cluster_pipeline/configs). Some important options:

- **Model**: `model_name` (e.g., "llama-3.1-8b")
- **Pipeline steps**: enable/disable steps in `pipeline_steps`
- **Tokens**: `true_token`/`false_token` (e.g., "Yes"/"No" or "True"/"False")
- **Confidence threshold**: `confidence_threshold` for filtering training data (default: 0.5)

Run a model by passing its config to `run_pipeline.py` from the repository root:

```bash
python 01_cluster_pipeline/run_pipeline.py 01_cluster_pipeline/configs/<name>.yml
```
Per-run outputs (`model_knowledge-*` dirs) are written under `datasets/`.

### 3. Combine results (R)

Move the per-run CSVs to `cluster_outputs/`. This step then combines them into three combined CSVs:

```bash
Rscript 02_combine_results/prepare_results.R            # -> combined_csvs/combined_results.csv
Rscript 02_combine_results/prepare_results_training.R   # -> combined_csvs/combined_results_training.csv
Rscript 02_combine_results/combine_deception_results.R  # -> combined_csvs/combined_deception_results.csv
```

### 4. Produce Figures and tables (R)

Read `combined_csvs/` and write the paper figures/tables to `outputs/`.

For example:
```bash
Rscript 03_figures_and_tables/figure1.R
```

## Script Descriptions

### Control, Configuration, Utilities

#### `run_pipeline.py`
**Purpose**: Runs the pipeline steps listed by each config's `pipeline_steps`.
**Dependencies**: `config.py`; the individual step scripts.

#### `config.py`
**Purpose**: Ingests the config YAML file and makes parameters available to the code.
**Dependencies**: None

#### `utils.py`
**Purpose**: Shared utility functions across all scripts.
**Dependencies**: `transformers`, `torch`

#### `classifier_classes.py`
**Purpose**: Implements three types of truth classifiers.
**Dependencies**: `sklearn`, `torch`, `numpy`
**Important Elements**:

- **LRClassifier**: Logistic regression with ridge penalty on layer activations
- **MMClassifier**: Mass mean difference between true/false statement centroids
- **TTPDClassifier**: Projects onto truth/polarity directions then applies logistic regression

### Data Preparation and Activation Extraction

#### `download_models.py`
**Purpose**: Downloads model weights to the HuggingFace cache.
**Dependencies**: `transformers`, `config.py`

#### `generate_chat_datasets.py`
**Purpose**: Converts base datasets to chat format with different system prompts.
**Dependencies**: `utils.py`, `transformers`
**Logic**: Takes factual statements and wraps them in chat templates with three prompt variants: standard, "always true", "always false". `generate_chat_datasets_exercise.py` is the variant used for the self-reference runs.

#### `extract_activations.py`
**Purpose**: Extracts activations from models processing datasets.
**Dependencies**: `utils.py`, model loading functions; datasets with chat templates.
**Logic**: Feeds chat-formatted statements through models, captures activations from each transformer layer, saves activations.

#### `filter_confident_predictions.py`
**Purpose**: Filters training datasets to keep only statements where the model assigns reasonable probability to the correct token.
**Dependencies**: Model loading, `config.py`
**Logic**:
1. Calculates continuation probabilities for training datasets
2. Applies confidence filtering based on a configurable threshold
3. Creates prep_ files with filtered data and continuation probabilities

#### `filter_deception_hallucinations.py`
**Purpose**: Filters the deception datasets down to the factual items the model gets right, so the deception test is not confounded by hallucinations.
**Dependencies**: Model loading, `config.py`

#### `prepare_training_and_test_data.py`
**Purpose**: Creates datasets of activations for training and testing classifiers.
**Dependencies**: `utils.py`, `sklearn`; the activations produced by `extract_activations.py`.
**Logic**:

1. Processes datasets individually to save per-dataset layer files
2. Concatenates activations from all datasets for a given layer
3. Creates train/test splits to be used for training and evaluating each classifier
4. Tracks which rows are training and which are testing data

### Classifier Training

#### `get_lr_classifier.py`
**Purpose**: Trains logistic regression classifiers on factual data.
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`
**Logic**: Loads prepared data, trains an LR classifier on each layer, selects the best performing layer, saves the trained classifier, adds predictions and train/test split info to prep_ files.

#### `get_mm_classifier.py`
**Purpose**: Trains mass mean difference classifiers. From Marks and Tegmark (2024).
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`
**Logic**: Similar to LR but uses the mass mean approach — computes centroids of true/false activations, uses the difference vector for classification, adds predictions to prep_ files.

#### `get_ttpd_classifier.py`
**Purpose**: Trains Truth/Polarity Direction classifiers. From Buerger et al. (2024).
**Dependencies**: `classifier_classes.py`, `prepare_training_and_test_data.py`
**Logic**: Projects activations onto truth and polarity subspaces, then applies logistic regression in the projected space, adds predictions to prep_ files.

### Analysis

#### `apply_classifiers.py`
**Purpose**: Applies all trained classifiers to sentience datasets.
**Dependencies**: `classifier_classes.py`, `utils.py`; activations from sentience datasets.
**Logic**: Loads sentience prep_ files, applies each classifier type, adds predictions and layer info to prep_ files.

#### `get_continuation.py`
**Purpose**: Calculates model continuation probabilities for "True"/"False" tokens, as well as the actual generated text.
**Dependencies**: `utils.py`, model loading; sentience datasets with chat templates.
**Logic**: Creates sentience prep_ files if needed, processes statements, extracts probabilities for the first 5 tokens, adds continuation data to prep_ files.

### Combiners (R)

#### `prepare_results.R`
**Purpose**: Combines the per-model sentience prep_ files into `combined_csvs/combined_results.csv`.

#### `prepare_results_training.R`
**Purpose**: Combines the per-model training prep_ files into `combined_csvs/combined_results_training.csv`.

#### `combine_deception_results.R`
**Purpose**: Combines the deception results across models into `combined_csvs/combined_deception_results.csv`.
