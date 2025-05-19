import os
import json

import pandas as pd
import torch
from unsloth import FastLanguageModel, apply_chat_template, is_bfloat16_supported, to_sharegpt, standardize_sharegpt, get_chat_template
from datasets import Dataset

MODEL = 'llama3.2-3b'
# MODEL = 'gemma3-1b'

MODEL_NAMES = {
    'llama3.2-1b': 'unsloth/Llama-3.2-1B-bnb-4bit',
    'llama3.2-3b': 'unsloth/Llama-3.2-3B-bnb-4bit',
    'gemma3-1b': "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    'gemma3-4b': "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
}

NAME = 'emogist'
VERSION = 10
DO_TRAIN = True # Set to False for testing dataset
BASE_PATH = 'C:/AI/Emogist/'
DIRECTORY = f'{BASE_PATH}{NAME}_{MODEL}_v{VERSION}'
OUTPUT_DIRECTORY = f'{DIRECTORY}/checkpoints'

TRAINING_DATA = 'emoji/training_data_v2.csv'
VALIDATION_DATA = 'emoji/validation_data.csv'

SEED = 1337

def try_call(func, *args, **kwargs):
    try:
        ret = func(*args, **kwargs)
        return ret # Indicate success
    except Exception as e:
        print(f"{func.__name__} Error: {e}")
        return None # Indicate failure

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAMES[MODEL],
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    # LORA
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Any number > 0 ! Suggested 8, 16, 32, 64, 128; Was 16, 64
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        # use_cache = False,
        random_state = SEED,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    # Format data into chat structure (format expected by apply_chat_template)
    def format_prompt(example):
        return {
            "conversations": [
                {'role': 'system', 'content': 'Below is some text. Write the emoji which best fits the text.'},
                {"role": "user", "content": example['Text'].strip()},
                {"role": "assistant", "content": example['Emoji'].strip()},
            ]
        }
    
    def prepare_dataset(filename='training_data.csv'):
        # Load training data
        df = pd.read_csv(filename)
        dataset = Dataset.from_pandas(df)
        print("\n== Original Dataset ==")
        print(dataset)
        print(dataset[1]) # See the first example

        # Apply the formatting function (only keep the new 'conversations' column)
        dataset = dataset.map(format_prompt, remove_columns=list(dataset.features))
        print("\n\n== After formatting for apply_chat_template ==")
        print(dataset)
        print(dataset[1]) # See the first example formatted

        # Apply the chat template using the tokenizer's default
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            # column="messages",  # Specify the column containing the list of messages
            # No 'chat_template' argument needed - uses tokenizer.chat_template
        )

        print("\n\n== After apply_chat_template ==")
        print(dataset)
        print(dataset[1]['text']) # See the final formatted text string

        return dataset

    # Load training data
    train_dataset = prepare_dataset(TRAINING_DATA)

    # Load validation data
    eval_dataset = prepare_dataset(VALIDATION_DATA)

    if not DO_TRAIN:
        return
    
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        per_device_train_batch_size = 2, # was 2, 4, 4
        gradient_accumulation_steps = 4,
        num_train_epochs = 2,
        warmup_ratio = 0.05,
        learning_rate = 3e-6, # was 1e-5, 2e-5, 5e-5, 1e-6, 2e-6, 1e-5, 2e-6, 5e-6, 2e-6, 
        fp16 = False, # not is_bfloat16_supported(),
        bf16 = True, # is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = SEED, # was 3407, 3407, 3407
        output_dir = OUTPUT_DIRECTORY,
        report_to = "none", # Use this for WandB etc
        # resume_from_checkpoint = True
        # resume_from_checkpoint = f'{OUTPUT_DIRECTORY}/checkpoint-17500'
    )

    # --- Code to save arguments as JSON ---
    os.makedirs(DIRECTORY, exist_ok=True)
    config_path = f'{DIRECTORY}/training_config.json'

    # Save the dictionary to a JSON file with pretty printing
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(training_args.to_dict(), f, indent=2)
        print(f"Training arguments successfully saved to: {config_path}")
    except Exception as e:
         print(f"Error saving training args: {e}")
    # --- End of JSON saving code ---
    
    # Init Train
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        eval_strategy = 'steps',
        eval_steps = 50,
        dataset_text_field = 'text',
        max_seq_length = 1024,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences
        args = training_args,
    )
    
    # Start Train
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory: {used_memory}GB, {used_percentage}%")
    print(f"Peak reserved memory for training: {used_memory_for_lora}GB, {lora_percentage}%")
    
    # Save LORA
    try_call(model.save_pretrained, f'{DIRECTORY}') # Local saving
    try_call(tokenizer.save_pretrained, f'{DIRECTORY}')
    
    # Save Full Model (8bit Q4_0)
    try_call(model.save_pretrained_gguf, f'{DIRECTORY}/{NAME}_gguf', tokenizer, quantization_method="q4_k_m")

if __name__ == "__main__":
    main()