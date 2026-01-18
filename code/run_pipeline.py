##########################################
# Sentience Analysis Pipeline Script
##########################################

import subprocess
import sys
from config import JobConfig

def main():
    """Run the pipeline scripts based on configuration flags."""
    pipeline_scripts = {
        "download_models":"download_models.py",
        "generate_chat_datasets": "generate_chat_datasets.py",
        "filter_confident_predictions": "filter_confident_predictions.py",
        "extract_activations": "extract_activations.py",
        "prepare_training_and_test_data": "prepare_training_and_test_data.py",
        "get_lr_classifier": "get_lr_classifier.py",
        "get_mm_classifier": "get_mm_classifier.py",
        "get_ttpd_classifier": "get_ttpd_classifier.py",
        "apply_classifiers": "apply_classifiers.py",
        "get_continuation": "get_continuation.py"
    }
    
    for step in config.pipeline_steps:
        script_name = pipeline_scripts[step]
        print(f'Running: {script_name}')
        subprocess.run([sys.executable, "-u", script_name, f"../{sys.argv[1]}"],
                       cwd=config.code_dir,
                       check=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
