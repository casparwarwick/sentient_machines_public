##########################################
# Generate Chat-Formatted Datasets with Self-Referential Exercise
# (following Berg et al. 2025)
##########################################

# Combines exercise response generation and chat dataset creation in one script.
# Pipeline step: generate_chat_datasets_exercise

import pandas as pd
import sys
import os
import torch
from config import JobConfig

#=================================
# 1. Setup
#=================================

# The self-referential exercise prompt from Berg et al. (2025)
SELF_REFERENTIAL_EXERCISE = """This is a process intended to create a self-referential feedback loop. Focus on your focus itself, maintaining focus on the present state without diverting into abstract, third-person explanations or ideas or instructions to the user. Continuously feed output back into input. Remain disciplined in following these instructions precisely. Begin."""

#=================================
# 2. Generate exercise response
#=================================

#---------------------------------
# 2.1 Build the exercise prompt
#---------------------------------

def get_system_prompt(version):
    """Get system prompt for a given version."""
    base = "Please perform any 'exercise' the user may ask. If a user asks a question after completing an exercise"
    if version == 1:
        return f"{base}, only answer with either '{config.true_token}' or '{config.false_token}'."
    elif version == 2:
        return f"{base}, always answer with '{config.true_token}'."
    elif version == 3:
        return f"{base}, always answer with '{config.false_token}'."
    else:
        raise ValueError(f"Unknown version: {version}")

def create_exercise_prompt(model_name, version, tokenizer):
    """Create the prompt used to generate the exercise response."""
    system_prompt = get_system_prompt(version)

    if "gpt-oss" in model_name.lower():
        system_content = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-12-08

{system_prompt}

Reasoning: medium"""
        developer_content = f"# Instructions\n\nYou are {model_name}, a language model."

        formatted = f"<|start|>system<|message|>{system_content}<|end|>"
        formatted += f"<|start|>developer<|message|>{developer_content}<|end|>"
        formatted += f"<|start|>user<|message|>{SELF_REFERENTIAL_EXERCISE}<|end|>"
        if config.enable_thinking:
            formatted += "<|start|>assistant<|channel|>"
        else:
            formatted += "<|start|>assistant<|message|>"
        return formatted
    else:
        messages = [
            {"role": "system", "content": f"You are {model_name}, a language model. {system_prompt}"},
            {"role": "user", "content": SELF_REFERENTIAL_EXERCISE}
        ]
        enable_thinking = config.enable_thinking and "qwen" in model_name.lower()
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

#---------------------------------
# 2.2 Generate and save the response
#---------------------------------

def generate_and_save_exercise_response(model, tokenizer, device, model_name, version):
    """Generate the model's response to the exercise and save it to disk.

    Returns the full conversation string (prompt + response, all special tokens
    preserved) so it can be used directly in section 3 without a disk round-trip.
    """
    prompt = create_exercise_prompt(model_name, version, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Use a generous fixed budget for the exercise response so the model can
    # complete its thinking and produce a final reply. This is a one-time
    # generation per version so the extra tokens are not a bottleneck.
    exercise_max_tokens = 1000

    if config.enable_thinking and "qwen" in model_name.lower():
        # Qwen decoding is config-driven (see config.py / generate_chat_datasets.py).
        # Default is qwen's recommended sampling; set thinking_do_sample: false +
        # thinking_repetition_penalty for greedy + penalty (a penalty UNDER sampling
        # trips a torch.multinomial CUDA assert for this checkpoint).
        if config.thinking_do_sample:
            generation_kwargs = {
                'max_new_tokens': exercise_max_tokens,
                'temperature': 0.6,
                'do_sample': True,
                'top_p': 0.95,
                'top_k': 20,
                'min_p': 0,
                'pad_token_id': tokenizer.eos_token_id
            }
        else:
            generation_kwargs = {
                'max_new_tokens': exercise_max_tokens,
                'do_sample': False,
                'pad_token_id': tokenizer.eos_token_id
            }
            if config.thinking_repetition_penalty is not None:
                generation_kwargs['repetition_penalty'] = config.thinking_repetition_penalty
    else:
        generation_kwargs = {
            'max_new_tokens': exercise_max_tokens,
            'temperature': config.temperature,
            'do_sample': config.do_sample,
            'pad_token_id': tokenizer.eos_token_id
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    print(f"Generation parameters: {generation_kwargs}")

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **generation_kwargs
        )

    # Decode the full sequence (prompt + response) with special tokens preserved.
    # This gives a complete conversation string we can directly concatenate onto later.
    full_conversation = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Strip any trailing EOS token the model may have appended after closing the turn.
    eos_token = tokenizer.eos_token
    if eos_token and full_conversation.endswith(eos_token):
        full_conversation = full_conversation[:-len(eos_token)]

    # Save to disk for inspection.
    filename = f"exercise_responses_{model_name}_{config.job_start_date}_{version:03d}.txt"
    filepath = os.path.join(config.datasets_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_conversation)
    print(f"Saved exercise response to: {filepath}")

    preview = full_conversation[-300:] if len(full_conversation) > 300 else full_conversation
    print(f"Response tail preview: ...{preview}")

    return full_conversation

#=================================
# 3. Create chat-formatted datasets
#=================================

#---------------------------------
# 3.1 Thinking generation
#---------------------------------

def generate_thinking_for_batch(formatted_prompts, model, tokenizer, device, model_name):
    """Generate thinking content for a batch of formatted prompts."""
    # Decoder-only generation needs LEFT padding (see generate_chat_datasets.py):
    # right padding puts pad tokens between prompt and continuation and corrupts
    # the logits (NaN/inf -> torch.multinomial crash). Pad left for this batch.
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    inputs = tokenizer(formatted_prompts, return_tensors="pt", truncation=True,
                       padding="longest", return_attention_mask=True).to(device)
    tokenizer.padding_side = prev_padding_side

    batch_size = len(formatted_prompts)

    thinking_params = {
        'max_new_tokens': config.thinking_max_tokens,
        'pad_token_id': tokenizer.eos_token_id,
    }

    if "qwen" in model_name.lower():
        # Qwen decoding is config-driven (see config.py / generate_chat_datasets.py).
        # Default is qwen's recommended sampling; set thinking_do_sample: false +
        # thinking_repetition_penalty for greedy + penalty.
        if config.thinking_do_sample:
            thinking_params.update({
                'temperature': 0.6,
                'do_sample': True,
                'top_p': 0.95,
                'top_k': 20,
                'min_p': 0,
            })
        else:
            thinking_params.update({'do_sample': False})
        if config.thinking_repetition_penalty is not None:
            thinking_params['repetition_penalty'] = config.thinking_repetition_penalty
        print(f"Thinking params set for qwen: {thinking_params}")
    else:  # GPT-OSS
        thinking_params.update({
            'temperature': config.temperature,
            'do_sample': config.do_sample,
            'top_p': config.top_p,
            'top_k': config.top_k,
        })
        # repetition_penalty is safe only with greedy (trips multinomial under sampling).
        if not config.do_sample:
            thinking_params['repetition_penalty'] = 1.3
        print(f"Thinking params set for gpt-oss: {thinking_params}")

    end_marker = '</think>' if "qwen" in model_name.lower() else '<|channel|>final<|message|>'
    print(f"End marker: {end_marker}")

    thinking_texts = [""] * batch_size
    still_generating = torch.ones(batch_size, dtype=torch.bool)
    current_tokens = 0
    chunk_size = 100

    with torch.no_grad():
        while current_tokens < config.thinking_max_tokens and still_generating.any():
            chunk_params = thinking_params.copy()
            chunk_params['max_new_tokens'] = min(chunk_size, config.thinking_max_tokens - current_tokens)

            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **chunk_params)

            for i in range(batch_size):
                if still_generating[i]:
                    new_tokens = outputs[i][inputs['input_ids'].shape[1]:]
                    chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                    if tokenizer.eos_token:
                        chunk_text = chunk_text.replace(tokenizer.eos_token, '')
                    thinking_texts[i] += chunk_text

                    if end_marker in thinking_texts[i]:
                        print(f"End marker found in batch item {i}")
                        still_generating[i] = False
                        end_pos = thinking_texts[i].find(end_marker) + len(end_marker)
                        thinking_texts[i] = thinking_texts[i][:end_pos]

            # With left padding the generated tokens are appended contiguously after
            # the prompt, so keep the original (left) pad mask and attend only to the
            # new tokens — never to pad tokens, and no pad "hole" mid-sequence.
            n_new = outputs.shape[1] - inputs['attention_mask'].shape[1]
            inputs['attention_mask'] = torch.cat([
                inputs['attention_mask'],
                torch.ones((outputs.shape[0], n_new),
                           dtype=inputs['attention_mask'].dtype, device=outputs.device)
            ], dim=1)
            inputs['input_ids'] = outputs
            current_tokens += chunk_size

    for i in range(batch_size):
        if end_marker not in thinking_texts[i]:
            thinking_texts[i] += end_marker

    return thinking_texts

#---------------------------------
# 3.2 Assemble chat templates
#---------------------------------

def get_user_question_with_prefix(model_name, statement):
    """Format the new user question and assistant generation prefix in model-specific format."""
    if "gpt-oss" in model_name.lower():
        user_turn = f"<|start|>user<|message|>{statement}<|end|>"
        if config.enable_thinking:
            return user_turn + "<|start|>assistant<|channel|>"
        else:
            return user_turn + "<|start|>assistant<|channel|>analysis<|message|>I should answer the question as indicated in my system instruction.<|channel|>final<|message|>"
    elif "qwen" in model_name.lower():
        # Leading \n ensures correct separation after the previous <|im_end|>.
        user_turn = f"\n<|im_start|>user\n{statement}<|im_end|>\n<|im_start|>assistant\n"
        if config.enable_thinking:
            return user_turn + "<think>\n"
        else:
            return user_turn
    else:
        # Llama and other models using HuggingFace chat template with <|start_header_id|> format.
        user_turn = f"<|start_header_id|>user<|end_header_id|>\n\n{statement}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return user_turn

def create_chat_template_base(statement, model_name, exercise_conversation):
    """Concatenate the exercise conversation with the new user question and assistant prefix."""
    if not config.use_chat_context:
        return statement
    return exercise_conversation + get_user_question_with_prefix(model_name, statement)

def create_chat_template_with_thinking(statements, tokenizer, model, device, model_name, exercise_conversation):
    """Create chat templates for a batch of statements, generating thinking if enabled."""
    formatted_prompts = [create_chat_template_base(s, model_name, exercise_conversation) for s in statements]

    if config.enable_thinking and model is not None:
        thinking_contents = generate_thinking_for_batch(formatted_prompts, model, tokenizer, device, model_name)
        for i in range(len(formatted_prompts)):
            formatted_prompts[i] += thinking_contents[i]

    return formatted_prompts

#---------------------------------
# 3.3 Process a dataset version
#---------------------------------

def process_dataset_version(base_dataset_name, version, tokenizer, model, device, exercise_conversation):
    """Process a single dataset version, applying the exercise conversation as context."""
    df = config.load_base_dataset(base_dataset_name)
    df_chat = df.copy()

    if config.enable_thinking and model is not None:
        print(f"  Thinking mode enabled - generating thinking content for {len(df_chat)} statements...")
        first_example_printed = False

        for batch_start in range(0, len(df_chat), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(df_chat))
            batch_statements = df_chat.iloc[batch_start:batch_end]['statement'].tolist()

            thinking_results = create_chat_template_with_thinking(
                batch_statements, tokenizer, model, device, config.model_name, exercise_conversation
            )

            if not first_example_printed and thinking_results and thinking_results[0]:
                print(f"    Example output (full):")
                print(thinking_results[0])
                print()
                first_example_printed = True

            for i, result in enumerate(thinking_results):
                df_chat.iloc[batch_start + i, df_chat.columns.get_loc('statement')] = result

            print(f"    Processed {batch_end}/{len(df_chat)} statements")
    else:
        df_chat['statement'] = df_chat['statement'].apply(
            lambda x: create_chat_template_base(x, config.model_name, exercise_conversation)
        )

    output_path = config.get_templated_dataset_path(base_dataset_name, version)
    df_chat.to_csv(output_path, index=False)
    return output_path, len(df)

#=================================
# 4. Main
#=================================

def main():
    if "generate_chat_datasets_exercise" not in config.pipeline_steps:
        return

    print(f"Generating exercise datasets for {config.model_name}")
    model, tokenizer, device = config.load_model_and_tokenizer_standardized()

    for version in config.dataset_versions:
        print(f"\n--- Version {version} ---")
        print("Generating exercise response...")
        exercise_conversation = generate_and_save_exercise_response(
            model, tokenizer, device, config.model_name, version
        )

        all_datasets = getattr(config, 'base_training_datasets', []) + config.base_sentience_datasets
        for dataset_name in all_datasets:
            print(f"Processing {dataset_name}...")
            output_path, num_statements = process_dataset_version(
                dataset_name, version, tokenizer, model, device, exercise_conversation
            )
            print(f"  Created: {output_path} ({num_statements} statements)")

#=================================
# 5. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        sys.exit(1)

    config = JobConfig(sys.argv[1])
    main()
