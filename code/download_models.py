#!/usr/bin/env python3

from huggingface_hub import snapshot_download
import os
import sys
from config import JobConfig

# Load huggingface credentials from .env
from dotenv import load_dotenv

load_dotenv()

def get_hf_model_name(model_name):
    """Map local model names to HuggingFace repository names."""
    model_map = {
        # GPT-OSS models
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        
        # Llama models
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct", 
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
        
        # Qwen models
        "qwen2-0.5b": "Qwen/Qwen2-0.5B",
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-8b": "Qwen/Qwen3-8B", 
        "qwen3-14b": "Qwen/Qwen3-14B",
        "qwen3-32b": "Qwen/Qwen3-32B",
        
        # Gemma models
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "gemma-2-27b-it": "google/gemma-2-27b-it",

        # Other models
        "phi-4": "microsoft/phi-4",
    }
    
    return model_map.get(model_name.lower(), model_name)

def get_ignore_patterns(hf_model_name):
    """Get ignore patterns for specific models."""
    if "meta-llama" in hf_model_name:
        return ["original/*"]
    elif "gpt-oss" in hf_model_name:
        return ["original/*", "metal/*"]
    return None

def download_model():
    """Download the model specified in the config."""
    if "download_models" not in config.pipeline_steps:
        return
    
    # Create models directory if it doesn't exist
    os.makedirs(config.models_dir, exist_ok=True)

    model_name = config.model_name

    # Get HuggingFace model name
    hf_model_name = get_hf_model_name(model_name)
    ignore_patterns = get_ignore_patterns(hf_model_name)
    
    local_dir = config.get_model_path()
    cache_dir = f"{config.models_dir}/.cache"
    
    # Download the model
    print(f"Downloading {hf_model_name} to {local_dir}")

    # Setting cache_dir in snapshot_download doesn't seem to move the xet cache.
    os.environ["HF_XET_CACHE"] = "models/.cache/xet"
    
    try:
        snapshot_download(
            repo_id=hf_model_name,
            local_dir=local_dir,
            cache_dir=cache_dir,
            ignore_patterns=ignore_patterns
        )
        print(f"Successfully downloaded {hf_model_name}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def main():
    """Main function."""
    download_model()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
