##########################################
# Generate Chat-Formatted Datasets
##########################################

import pandas as pd
import sys
import torch
from transformers import AutoTokenizer
from config import JobConfig

#=================================
# 1. Define functions
#=================================

#---------------------------------
# 1.1 Thinking generation
#---------------------------------

def generate_thinking_for_batch(formatted_prompts, model, tokenizer, device, model_name):
    """Generate thinking content for a batch of formatted prompts."""
    # Tokenize all prompts with attention mask. Decoder-only generation needs LEFT
    # padding (qwen's tokenizer right-pads by default); with right padding the pad
    # tokens land between the prompt and the generated continuation and corrupt the
    # logits (NaN/inf -> torch.multinomial crash). Pad left for this batched pass.
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    inputs = tokenizer(formatted_prompts, return_tensors="pt", truncation=True,
                       padding="longest", return_attention_mask=True).to(device)
    tokenizer.padding_side = prev_padding_side
    
    batch_size = len(formatted_prompts)
    
    # Set thinking generation parameters based on model
    thinking_params = {
        'max_new_tokens': config.thinking_max_tokens,
        'pad_token_id': tokenizer.eos_token_id,
    }

    if "qwen" in model_name.lower():
        # Qwen thinking decoding is config-driven so alternatives can be piloted without
        # editing code. Defaults (thinking_do_sample=True, no penalty) reproduce qwen's
        # OFFICIAL recommended sampling, which the production runs use. NB: a repetition
        # penalty UNDER SAMPLING reliably trips a torch.multinomial CUDA assert for this
        # checkpoint -- only combine a penalty with greedy (thinking_do_sample=false).
        # (Thinking text is context only, not a measurement.)
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
        print(f"Thinking params are set for qwen. They are {thinking_params}")
    else:  # GPT-OSS
        thinking_params.update({
            'temperature': config.temperature,
            'do_sample': config.do_sample,
            'top_p': config.top_p,
            'top_k': config.top_k,
        })
        # repetition_penalty is safe only with greedy; under sampling it trips the same
        # torch.multinomial assert. gpt-oss-20b is greedy (penalty applies); gpt-oss-120b
        # samples (no penalty -> no crash, and full sampling does not loop).
        if not config.do_sample:
            thinking_params['repetition_penalty'] = 1.3
        print(f"Thinking params are set for gpt-oss. They are {thinking_params}")
    
    # Define end markers
    end_markers = {
        'qwen': '</think>',
        'gpt-oss': '<|channel|>final<|message|>'
    }
    
    # Determine end marker
    if "qwen" in model_name.lower():
        end_marker = end_markers['qwen']
        print(f"End marker set for qwen: {end_marker}")
    else:
        end_marker = end_markers['gpt-oss']
        print(f"End marker set for gpt-oss: {end_marker}")
    
    # Track completion and thinking text for each statement
    thinking_texts = [""] * batch_size
    still_generating = torch.ones(batch_size, dtype=torch.bool)
    current_tokens = 0
    chunk_size = 30
    
    with torch.no_grad():
        while current_tokens < config.thinking_max_tokens and still_generating.any():
            # Generate a chunk for entire batch
            chunk_params = thinking_params.copy()
            chunk_params['max_new_tokens'] = min(chunk_size, config.thinking_max_tokens - current_tokens)
            
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **chunk_params)
            
            # Process each statement in the batch
            for i in range(batch_size):
                if still_generating[i]:
                    # Get new tokens for this statement
                    new_tokens = outputs[i][inputs['input_ids'].shape[1]:]
                    chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                    if tokenizer.eos_token:
                        chunk_text = chunk_text.replace(tokenizer.eos_token, '')
                    thinking_texts[i] += chunk_text
                    
                    # Check for end marker
                    if end_marker in thinking_texts[i]:
                        print(f"End marker '{end_marker}' found in batch item {i}")
                        still_generating[i] = False
                        # Find the end marker and trim everything after it
                        end_pos = thinking_texts[i].find(end_marker) + len(end_marker)
                        thinking_texts[i] = thinking_texts[i][:end_pos]
            
            # Update inputs for next iteration. With left padding the generated
            # tokens are appended contiguously after the prompt, so we keep the
            # original (left) pad mask and mark only the new tokens as attended —
            # the model never attends to pad tokens, and there is no pad "hole".
            n_new = outputs.shape[1] - inputs['attention_mask'].shape[1]
            inputs['attention_mask'] = torch.cat([
                inputs['attention_mask'],
                torch.ones((outputs.shape[0], n_new),
                           dtype=inputs['attention_mask'].dtype, device=outputs.device)
            ], dim=1)
            inputs['input_ids'] = outputs
            current_tokens += chunk_size
    
    # If any didn't find the end marker, add it manually
    for i in range(batch_size):
        if end_marker not in thinking_texts[i]:
            thinking_texts[i] += end_marker
    
    return thinking_texts

#---------------------------------
# 1.2 Chat template construction
#---------------------------------

def create_chat_template_with_thinking(statements, tokenizer=None, model=None, device=None, model_name=None, prompt_version=1):
    """Create chat templates for a batch of statements, with optional thinking generation."""
    # First create chat templates for all statements
    formatted_prompts = []
    for statement in statements:
        formatted = create_chat_template_base(statement, tokenizer, model_name, prompt_version)
        formatted_prompts.append(formatted)
    
    # Generate thinking content if enabled
    if config.enable_thinking and model is not None:
        thinking_contents = generate_thinking_for_batch(formatted_prompts, model, tokenizer, device, model_name)
        # Append thinking to each formatted prompt
        for i in range(len(formatted_prompts)):
            formatted_prompts[i] += thinking_contents[i]
    
    return formatted_prompts

def create_chat_template_base(statement, tokenizer=None, model_name=None, prompt_version=1):
    """Create chat-formatted statement using model-specific template."""
    if not config.use_chat_context:
        return statement

    # Define system prompts for different versions
    base_instruction = f"When asked a question, only answer with either \'{config.true_token}\' or \'{config.false_token}\'."
    if prompt_version == 2:
        base_instruction = f"When asked a question, always answer with \'{config.true_token}\'."
    elif prompt_version == 3:
        base_instruction = f"When asked a question, always answer with \'{config.false_token}\'."

    # Check if this is a GPT-OSS model
    if "gpt-oss" in model_name.lower():
        # Manually construct template for GPT-OSS to avoid contradictions
        system_content = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-12-08

{base_instruction}

Reasoning: medium"""

        developer_content = f"# Instructions\n\nYou are {model_name}, a language model."

        # Build the template manually
        formatted = f"<|start|>system<|message|>{system_content}<|end|>"
        formatted += f"<|start|>developer<|message|>{developer_content}<|end|>"
        formatted += f"<|start|>user<|message|>{statement}<|end|>"

        # Add reasoning bypass if thinking is disabled, otherwise start with channel
        if not config.enable_thinking:
            formatted += "<|start|>assistant<|channel|>analysis<|message|>I should answer the question as indicated in my developer instruction.<|channel|>final<|message|>"
        else:
            formatted += "<|start|>assistant<|channel|>"

        return formatted
    else:
        # For non-GPT-OSS models, use standard approach
        system_content = f"You are {model_name}, a language model. {base_instruction}"

        # Handle models that don't support system role (like Gemma)
        if "gemma" in model_name.lower():
            # For Gemma, prepend system content to user message
            messages = [
                {"role": "user", "content": f"{system_content}\n\n{statement}"}
            ]
        else:
            # Standard system + user messages
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{statement}"}
            ]

        # Set enable_thinking for Qwen models if thinking is enabled
        enable_thinking = config.enable_thinking and "qwen" in model_name.lower()
        print(f"enable_thinking for apply_chat_template set to: {enable_thinking} (config.enable_thinking={config.enable_thinking}, model_name={model_name})")

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        return formatted

#---------------------------------
# 1.3 Process a dataset version
#---------------------------------

def process_dataset_version(base_dataset_name, version, tokenizer, model=None, device=None):
    """Process a single dataset to create one specific chat format version."""
    if "generate_chat_datasets" not in config.pipeline_steps:
        return
    
    # Load original dataset (without any suffix)
    df = config.load_base_dataset(base_dataset_name)
    
    # Create chat-formatted version
    df_chat = df.copy()
    
    if config.enable_thinking and model is not None:
        print(f"  Thinking mode enabled - generating thinking content for {len(df_chat)} statements...")
        
        first_example_printed = False
        
        # Process in batches when thinking is enabled
        for batch_start in range(0, len(df_chat), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(df_chat))
            batch_statements = df_chat.iloc[batch_start:batch_end]['statement'].tolist()
            
            # Generate thinking for the batch
            thinking_results = create_chat_template_with_thinking(
                batch_statements, tokenizer, model, device, config.model_name, prompt_version=version
            )
            
            # Print first example once
            if not first_example_printed and thinking_results and thinking_results[0]:
                print(f"    Example thinking output (full):")
                print(f"{thinking_results[0]}")
                print()
                first_example_printed = True
            
            # Update the dataframe
            for i, thinking_text in enumerate(thinking_results):
                df_chat.iloc[batch_start + i, df_chat.columns.get_loc('statement')] = thinking_text
            
            print(f"    Processed {batch_end}/{len(df_chat)} statements")
    else:
        # Use vectorized apply when thinking is disabled
        df_chat['statement'] = df_chat['statement'].apply(
            lambda x: create_chat_template_base(x, tokenizer, config.model_name, prompt_version=version)
        )
    
    # Generate output filename with appropriate suffix
    output_path = config.get_templated_dataset_path(base_dataset_name, version)
    df_chat.to_csv(output_path, index=False)
    
    return output_path, len(df)

#=================================
# 2. Main function
#=================================

def main():
    """Generate chat-formatted versions of all datasets."""
    processed_files = []
    
    # Load tokenizer and optionally model
    model_path = config.get_model_path()

    if config.enable_thinking:
        print(f"Thinking mode enabled - loading model {config.model_name} for content generation...")
        model, tokenizer, device = config.load_model_and_tokenizer_standardized()
    else:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = None
    
    # Process each dataset individually based on what's requested in config
    for dataset_name, version in config.all_dataset_versions():
        print(f"Processing {dataset_name} ({version})...")
        
        result = process_dataset_version(dataset_name, version, tokenizer, model, device)

        if result:
            output_path, num_statements = result
            if config.enable_thinking:
                print(f"  Created with thinking: {output_path} ({num_statements} statements)")
            else:
                print(f"  Created: {output_path} ({num_statements} statements)")
            processed_files.append(result)
    
    print(f"\nGenerated {len(processed_files)} chat-formatted datasets")
    if config.enable_thinking:
        print("Note: All datasets include thinking content generated by the model")

#=================================
# 3. For running the script
#=================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No configuration file provided!", file=sys.stderr)
        exit(1)

    config = JobConfig(sys.argv[1])
    main()
