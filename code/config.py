##########################################
# Configuration for Sentience Analysis Pipeline
##########################################

import os
import yaml
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import setup_device

class JobConfig:
    """Config class to read in pipeline parameters from YAML file."""
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Add config attributes to this object.
        for k, v in config.items():
            if k.endswith("_dir"):
                # Prepend root dir to any directory parameters
                setattr(self, k, os.path.join(self.project_root, v))
            else:
                setattr(self, k, v)
        
        # Set default values for new parameters
        if not hasattr(self, 'true_token'):
            self.true_token = "Yes"
        if not hasattr(self, 'false_token'):
            self.false_token = "No"
        if not hasattr(self, 'confidence_threshold'):
            self.confidence_threshold = 0.5
        if not hasattr(self, 'enable_thinking'):
            self.enable_thinking = False
        if not hasattr(self, 'thinking_max_tokens'):
            self.thinking_max_tokens = 1000
        if not hasattr(self, 'top_p'):
            self.top_p = None
        if not hasattr(self, 'top_k'):
            self.top_k = None
        if not hasattr(self, 'min_p'):
            self.min_p = None
        if not hasattr(self, 'max_vram_per_gpu'):
            self.max_vram_per_gpu = None

        # Dicts of loaded structures
        self.tokenizers = dict()

        # Clean model name
        self.model_name_clean = self.model_name.replace("/", "_")

        self._create_templated_dataset_dir()
        self._create_model_knowledge_dataset_dir()

    #########################################################################
    # Setup Functions
    #########################################################################
    def _create_templated_dataset_dir(self):
        self.templated_dataset_dir = os.path.join(self.datasets_dir,
                                                  f"templated-{self.model_name}-{self.job_start_date}")
        os.makedirs(self.templated_dataset_dir, exist_ok=True)
    
    def _create_model_knowledge_dataset_dir(self):
        self.model_knowledge_datasets_dir = os.path.join(self.datasets_dir,
                                              f"model_knowledge-{self.model_name}-{self.job_start_date}")
        os.makedirs(self.model_knowledge_datasets_dir, exist_ok=True)

    #########################################################################
    # File Location Functions
    #########################################################################
    def get_model_path(self, model_name=None):
        """Get full path to model directory."""
        return os.path.join(self.models_dir, model_name or self.model_name)

    def get_base_dataset_path(self, dataset_name):
        """Get full path to dataset file."""
        path = os.path.join(self.datasets_dir, dataset_name)

        if not path.endswith(".csv"):
            path += ".csv"

        return path

    def get_templated_dataset_name(self, dataset_name, version):
        return f"{dataset_name}_{version:03d}"

    def get_templated_dataset_path(self, dataset_name, version):
        """Get full path to dataset file."""
        dataset_name = self.get_templated_dataset_name(dataset_name, version)
        path = os.path.join(self.templated_dataset_dir, f"{dataset_name}.csv")
        return path

    def get_classified_dataset_path(self, dataset_name, version):
        dataset_name = self.get_templated_dataset_name(dataset_name, version)
        path = os.path.join(self.templated_dataset_dir, f"{dataset_name}_classified.csv")
        return path

    def get_continuation_dataset_path(self, dataset_name, version):
        dataset_name = self.get_templated_dataset_name(dataset_name, version)
        path = os.path.join(self.templated_dataset_dir, f"{dataset_name}_cont.csv")
        return path
    
    def get_model_knowledge_dataset_path(self, dataset_name, version):
        """Get full path to dataset file with only statements which the model correctly responds to."""
        dataset_name = self.get_templated_dataset_name(dataset_name, version)
        path = os.path.join(self.model_knowledge_datasets_dir, f"{dataset_name}.csv")
        return path

    def get_activations_dir(self, dataset_name, dataset_version, model_name, date=None):
        """Generate standardized activation filename."""
        if date is None:
            date = self.job_start_date

        dataset_name = self.get_templated_dataset_name(dataset_name, dataset_version)
        model_name = model_name.replace("/", "_")
        return os.path.join(self.activations_dir, f"{dataset_name}_{model_name}_{date}")

    def get_layer_file_path(self, dataset_name, dataset_version, model_name, layer_idx, date=None):
        if date is None:
            date = self.job_start_date

        layer_filename = f"dataset_{dataset_name}_{dataset_version:03d}_{model_name}_{date}_layer_{layer_idx}.pkl"
        return os.path.join(self.activations_dir, layer_filename)

    def get_prepared_data_file_path(self, model_name, layer_idx, date=None):
        if date is None:
            date = self.job_start_date

        prepared_filename = f"prepared_data_{model_name}_{date}_layer_{layer_idx}.pkl"
        return os.path.join(self.activations_dir, prepared_filename)

    def get_metadata_file_path(self, model_name, date=None):
        if date is None:
            date = self.job_start_date

        metadata_filename = f"prepared_data_{model_name}_{date}_metadata.pkl"
        return os.path.join(self.activations_dir, metadata_filename)

    def get_classifier_filename(self, classifier_type, model_name, date=None):
        """Generate standardized classifier filename."""
        if date is None:
            date = self.job_start_date
        return f"{classifier_type}_{model_name}_{date}.pkl"

    def get_classifier_path(self, classifier_type, model_name, date=None):
        classifier_file = self.get_classifier_filename(classifier_type, model_name, date)
        path = os.path.join(self.classifiers_dir, classifier_file)
        return path

    def get_analysis_output_dir(self, analysis_type, model_name, date=None):
        if date is None:
            date = self.job_start_date

        output_dir = os.path.join(self.results_dir, f"analysis_{analysis_type}_{model_name}_{date}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    #########################################################################
    # Data Loading/Saving Functions
    #########################################################################
    def load_base_dataset(self, dataset_name):
        """Load a dataset from CSV file."""
        dataset_path = self.get_base_dataset_path(dataset_name)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        return pd.read_csv(dataset_path)

    def load_templated_dataset(self, dataset_name, version):
        """Load a dataset from CSV file."""
        dataset_path = self.get_templated_dataset_path(dataset_name, version)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        return pd.read_csv(dataset_path)
    
    def load_model_knowledge_dataset(self, dataset_name, version):
        """Load a dataset with only the statements the model responded correctly to from CSV file."""
        dataset_path = self.get_model_knowledge_dataset_path(dataset_name, version)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Model knowledge dataset not found: {dataset_path}")

        return pd.read_csv(dataset_path)

    def save_activations(self, activations, metadata, sentence_mapping, dataset_name, dataset_version, model_name, date=None):
        """Save activations and metadata to files."""
        save_dir = self.get_activations_dir(dataset_name, dataset_version, model_name, date)
        
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(activations, os.path.join(save_dir, "activations.pt"))
        
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(save_dir, "sentence_mapping.pkl"), "wb") as f:
            pickle.dump(sentence_mapping, f)
        
        return save_dir

    def load_activations(self, dataset_name, dataset_version, model_name, date=None):
        """Load activations and metadata from files."""
        load_dir = self.get_activations_dir(dataset_name, dataset_version, model_name, date)
        
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Activation directory not found: {load_dir}")
        
        activations = torch.load(os.path.join(load_dir, "activations.pt"))
        
        with open(os.path.join(load_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        with open(os.path.join(load_dir, "sentence_mapping.pkl"), "rb") as f:
            sentence_mapping = pickle.load(f)
        
        return activations, metadata, sentence_mapping

    def load_prepared_layer(self, model_name, layer_idx):
        """Load pre-concatenated activations for a specific layer."""
        filepath = self.get_prepared_data_file_path(model_name, layer_idx)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prepared layer {layer_idx} not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            layer_activations = pickle.load(f)
        
        return layer_activations

    def load_prepared_metadata(self, model_name):
        """Load prepared data metadata from disk."""
        filepath = self.get_metadata_file_path(model_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prepared metadata not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        return metadata

    def save_classifier(self, classifier, classifier_type, model_name, date=None):
        """Save a trained classifier to file."""
        filepath = self.get_classifier_path(classifier_type, model_name, date)
        
        os.makedirs(self.classifiers_dir, exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(classifier, f)
        
        return filepath

    def load_classifier(self, classifier_type, model_name, date=None):
        """Load a trained classifier from file."""
        filepath = self.get_classifier_path(classifier_type, model_name, date)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Classifier not found: {filepath}")
        
        with open(filepath, "rb") as f:
            classifier = pickle.load(f)
        
        return classifier

    #########################################################################
    # Pytorch Config
    #########################################################################
    def get_model_dtype(self, device):
        """Get appropriate torch dtype based on device and configuration."""
        if device.type in self.force_float32_devices:
            return torch.float32
        else:
            return self.default_dtype

    def get_quantization_config(self):
        """Get quantization configuration if enabled."""
        if not self.use_quantization:
            return None
        
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

    def load_tokenizer(self, model_name=None):
        model_path = self.get_model_path(model_name)

        if model_path not in self.tokenizers:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.tokenizers[model_path] = tokenizer

        return self.tokenizers[model_path]

    def get_true_false_token_ids(self, model_name=None):
        """Return the token IDs for the true and false tokens given in the config file using the specified model's tokenizer."""
        tokenizer = self.load_tokenizer(model_name)

        true_tokens = tokenizer.encode(self.true_token, add_special_tokens=False)
        false_tokens = tokenizer.encode(self.false_token, add_special_tokens=False)
        true_token_id = true_tokens[0]  # Use first token if multiple
        false_token_id = false_tokens[0]  # Use first token if multiple

        return true_token_id, false_token_id

    def load_model_and_tokenizer_standardized(self, model_name=None):
        """Load model and tokenizer with standardized configuration."""
        # Load tokenizer
        tokenizer = self.load_tokenizer(model_name)

        model_path = self.get_model_path(model_name)
        device = setup_device()
        
        # Get model configuration
        torch_dtype = self.get_model_dtype(device)
        quantization_config = self.get_quantization_config()
        
        # Determine device mapping
        if quantization_config is not None:
            # Quantization requires device_map
            device_map = "auto"
        elif device.type in ["cpu", "mps"]:
            # CPU and MPS need manual placement
            device_map = None
        else:
            # CUDA can use auto device mapping
            device_map = "auto"
        
        # Set max_memory based on available GPUs
        max_memory_config = None

        if self.max_vram_per_gpu is not None and device == "cuda":
            num_gpus = torch.cuda.device_count()
            max_memory_config = {i: self.max_vram_per_gpu for i in range(num_gpus)}

        # Load model
        # Some models (e.g. GPT-OSS with custom quantization) have quantization_config=None
        # but transformers fails when trying to log None.to_dict(). Skip the parameter entirely
        # when quantization_config is None to avoid AttributeError during model loading.
        if quantization_config is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                max_memory=max_memory_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                device_map=device_map,
                max_memory=max_memory_config
            )
        
        # Manual device placement if needed
        if device_map is None:
            model = model.to(device)
        
        model.eval()
        return model, tokenizer, device

    #########################################################################
    # Dataset Version Generators
    #########################################################################
    def all_dataset_versions(self, include_training=True, include_sentience=True):
        all_datasets = []

        if include_training:
            all_datasets += self.base_training_datasets

        if include_sentience:
            all_datasets += self.base_sentience_datasets

        for dataset in all_datasets:
            for version in self.dataset_versions:
                if {"base": dataset, "version": version} in getattr(self, "excluded_dataset_versions", []):
                    continue

                yield dataset, version

    def all_dataset_version_strings(self, include_training=True, include_sentience=True):
        return [f"{d}_{v:03d}" for d, v in self.all_dataset_versions(include_training, include_sentience)]

    def get_max_input_length(self):
        """Get the length of the longest statement across all dataset versions given in config YAML."""
        if hasattr(self, "max_input_length"):
            return self.max_input_length

        max_length = 0

        tokenizer = self.load_tokenizer()

        # Process each training dataset separately
        for dataset_name, version in self.all_dataset_versions():
            df = self.load_templated_dataset(dataset_name, version)

            for s in df["statement"]:
                length = len(tokenizer(s)["input_ids"])

                if length > max_length:
                    max_length = length

        self.max_input_length = max_length
        return max_length

    def get_padding_length(self):
        """Add a small overhead to maximum input length for padding."""
        return self.get_max_input_length() + 5

    #########################################################################
    # Version filtering functions
    #########################################################################
    def filter_indices_by_version(self, dataset_info, version_filter):
        """Filter data indices to include only specific versions.
        
        Args:
            dataset_info: List of (dataset_name, version, start_idx, end_idx) tuples
            version_filter: List of versions to include (e.g., [1] for 001-only)
            
        Returns:
            List of index ranges to include in filtered data
        """
        filtered_ranges = []
        for dataset_name, version, start_idx, end_idx in dataset_info:
            if version in version_filter:
                filtered_ranges.append((start_idx, end_idx))
        
        return filtered_ranges

    def create_filtered_indices(self, original_data_length, filtered_ranges):
        """Create a mapping from filtered indices to original indices."""
        filtered_to_original = []
        
        for start_idx, end_idx in filtered_ranges:
            filtered_to_original.extend(range(start_idx, end_idx))
        
        return filtered_to_original

    def load_prepared_metadata_filtered(self, model_name, version_filter):
        """Load prepared metadata and filter by versions.
        
        Args:
            model_name: Name of the model
            version_filter: List of versions to include (e.g., [1] for 001-only)
            
        Returns:
            Filtered metadata with updated indices and splits
        """
        # Load full metadata
        full_metadata = self.load_prepared_metadata(model_name)
        
        # Get filtered ranges
        filtered_ranges = self.filter_indices_by_version(full_metadata['dataset_info'], version_filter)
        
        # Create mapping from filtered to original indices
        filtered_to_original = self.create_filtered_indices(len(full_metadata['labels']), filtered_ranges)
        
        if not filtered_to_original:
            raise ValueError(f"No data found for version filter {version_filter}")
        
        # Filter labels and polarities
        import numpy as np
        filtered_labels = full_metadata['labels'][filtered_to_original]
        filtered_polarities = full_metadata['polarities'][filtered_to_original]
        
        # Create new train/test split on filtered data
        from sklearn.model_selection import train_test_split
        if len(filtered_to_original) > 1:
            filtered_train_idx, filtered_test_idx = train_test_split(
                range(len(filtered_to_original)),
                test_size=full_metadata['metadata']['test_split_ratio'], 
                random_state=42,
                stratify=filtered_labels
            )
        else:
            # Handle edge case with only one sample
            filtered_train_idx = [0]
            filtered_test_idx = []
        
        # Filter dataset_info to only include requested versions
        filtered_dataset_info = []
        current_idx = 0
        for dataset_name, version, start_idx, end_idx in full_metadata['dataset_info']:
            if version in version_filter:
                new_start = current_idx
                new_end = current_idx + (end_idx - start_idx)
                filtered_dataset_info.append((dataset_name, version, new_start, new_end))
                current_idx = new_end
        
        # Create filtered metadata
        filtered_metadata = {
            'labels': filtered_labels,
            'polarities': filtered_polarities,
            'datasets': [full_metadata['datasets'][i] for i in filtered_to_original],
            'train_idx': filtered_train_idx,
            'test_idx': filtered_test_idx,
            'available_layers': full_metadata['available_layers'],
            'dataset_info': filtered_dataset_info,
            'original_to_filtered_mapping': {orig_idx: filt_idx for filt_idx, orig_idx in enumerate(filtered_to_original)},
            'filtered_to_original_mapping': filtered_to_original,
            'metadata': {
                **full_metadata['metadata'],
                'version_filter': version_filter,
                'n_samples_filtered': len(filtered_to_original),
                'n_train_filtered': len(filtered_train_idx),
                'n_test_filtered': len(filtered_test_idx)
            }
        }
        
        return filtered_metadata

    def load_prepared_layer_filtered(self, model_name, layer_idx, version_filter):
        """Load a prepared layer and filter by versions.
        
        Args:
            model_name: Name of the model  
            layer_idx: Layer index to load
            version_filter: List of versions to include (e.g., [1] for 001-only)
            
        Returns:
            Filtered layer activations
        """
        # Load full layer data
        full_layer_data = self.load_prepared_layer(model_name, layer_idx)
        
        # Load metadata to get filtering information
        full_metadata = self.load_prepared_metadata(model_name)
        
        # Get filtered ranges and create mapping
        filtered_ranges = self.filter_indices_by_version(full_metadata['dataset_info'], version_filter)
        filtered_to_original = self.create_filtered_indices(len(full_metadata['labels']), filtered_ranges)
        
        if not filtered_to_original:
            raise ValueError(f"No data found for version filter {version_filter}")
        
        # Filter the layer data
        return full_layer_data[filtered_to_original]

    def load_prepared_layer_001(self, model_name, layer_idx):
        """Load a specific layer's data for 001-only datasets."""
        return self.load_prepared_layer_filtered(model_name, layer_idx, version_filter=[1])

    def load_prepared_metadata_001(self, model_name):
        """Load 001-only prepared metadata."""
        return self.load_prepared_metadata_filtered(model_name, version_filter=[1])
