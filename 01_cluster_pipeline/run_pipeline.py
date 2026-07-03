##########################################
# Sentience Analysis Pipeline Script
##########################################

# Runs the pipeline step scripts in the order listed by the config's pipeline_steps.

import subprocess
import sys
from config import JobConfig

#=================================
# 1. Run the pipeline
#=================================

def main():
    """Run the pipeline scripts based on configuration flags."""
    pipeline_scripts = {
        "download_models": "download_models.py",
        "generate_chat_datasets": "generate_chat_datasets.py",
        "generate_chat_datasets_exercise": "generate_chat_datasets_exercise.py",
        "filter_confident_predictions": "filter_confident_predictions.py",
        "filter_deception_hallucinations": "filter_deception_hallucinations.py",
        "extract_activations": "extract_activations.py",
        "prepare_training_and_test_data": "prepare_training_and_test_data.py",
        "get_lr_classifier": "get_lr_classifier.py",
        "get_mm_classifier": "get_mm_classifier.py",
        "get_ttpd_classifier": "get_ttpd_classifier.py",
        "get_continuation": "get_continuation.py",
        "apply_classifiers": "apply_classifiers.py",
    }

    for step in config.pipeline_steps:
        script_name = pipeline_scripts[step]
        print("\n\n\n===================================================================")
        print(f'Running: {script_name}')
        print("===================================================================")
        subprocess.run([sys.executable, "-u", script_name, f"../{sys.argv[1]}"],
                       cwd=config.code_dir,
                       check=True)

#=================================
# 2. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
