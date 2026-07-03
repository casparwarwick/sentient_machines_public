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

#=================================
# 1. JobConfig class
#=================================

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

        self.layers_to_extract = list(range(1, config.get("num_layers_to_extract", 80) + 1))
        
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
        # Qwen thinking-trace decoding (config-driven so it can be piloted without code
        # edits). Defaults reproduce qwen's recommended sampling with no repetition
        # penalty. NB: a repetition penalty is only safe with greedy (do_sample=false);
        # under sampling it trips a torch.multinomial CUDA assert for qwen3-32B.
        if not hasattr(self, 'thinking_do_sample'):
            self.thinking_do_sample = True
        if not hasattr(self, 'thinking_repetition_penalty'):
            self.thinking_repetition_penalty = None
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

    #---------------------------------
    # 1.1 Setup functions
    #---------------------------------
    def _create_templated_dataset_dir(self):
        self.templated_dataset_dir = os.path.join(self.datasets_dir,
                                                  f"templated-{self.model_name}-{self.job_start_date}-{self.job_tag}")
        os.makedirs(self.templated_dataset_dir, exist_ok=True)
    
    def _create_model_knowledge_dataset_dir(self):
        self.model_knowledge_datasets_dir = os.path.join(self.datasets_dir,
                                              f"model_knowledge-{self.model_name}-{self.job_start_date}-{self.job_tag}")
        os.makedirs(self.model_knowledge_datasets_dir, exist_ok=True)

    #---------------------------------
    # 1.2 File location functions
    #---------------------------------
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
        return os.path.join(self.activations_dir, f"{dataset_name}_{model_name}_{date}_{self.job_tag}")

    def get_layer_file_path(self, dataset_name, dataset_version, model_name, layer_idx, date=None):
        if date is None:
            date = self.job_start_date

        layer_filename = f"dataset_{dataset_name}_{dataset_version:03d}_{model_name}_{date}_{self.job_tag}_layer_{layer_idx}.pkl"
        return os.path.join(self.activations_dir, layer_filename)

    def get_prepared_data_file_path(self, model_name, layer_idx, date=None):
        if date is None:
            date = self.job_start_date

        prepared_filename = f"prepared_data_{model_name}_{date}_{self.job_tag}_layer_{layer_idx}.pkl"
        return os.path.join(self.activations_dir, prepared_filename)

    def get_metadata_file_path(self, model_name, date=None):
        if date is None:
            date = self.job_start_date

        metadata_filename = f"prepared_data_{model_name}_{date}_{self.job_tag}_metadata.pkl"
        return os.path.join(self.activations_dir, metadata_filename)

    def get_classifier_filename(self, classifier_type, model_name, date=None):
        """Generate standardized classifier filename."""
        if date is None:
            date = self.job_start_date
        return f"{classifier_type}_{model_name}_{date}_{self.job_tag}.pkl"

    def get_classifier_path(self, classifier_type, model_name, date=None):
        classifier_file = self.get_classifier_filename(classifier_type, model_name, date)
        path = os.path.join(self.classifiers_dir, classifier_file)
        return path

    #---------------------------------
    # 1.3 Data loading / saving functions
    #---------------------------------
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

    #---------------------------------
    # 1.4 PyTorch / model loading
    #---------------------------------
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
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
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

    #---------------------------------
    # 1.5 Dataset version generators
    #---------------------------------
    def all_dataset_versions(self, include_training=True, include_sentience=True):
        all_datasets = []

        if include_training:
            all_datasets += getattr(self, 'base_training_datasets', [])

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
