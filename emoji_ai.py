import os
import re
import json
from ollama import Client

# Step 2

ollama_client = Client()

def ollama(input, model, system=None, temperature=0.7, num_predict=100):
    gen = ollama_client.generate(
        model=model,
        system=system,
        prompt=input,
        options={
            'temperature': temperature,
            'num_predict': num_predict
        }
    )
    tokens = (gen.eval_count or 0) + (gen.prompt_eval_count or 0)//4
    return gen.response or '', tokens

# Step 7

def parent_name(directory_path):
    # 'C:/AI/Emogist/emogist_llama3.2-1b_v0/checkpoints' -> 'emogist_llama3.2-1b_v0'
    parent_dir = os.path.dirname(directory_path)
    return os.path.basename(parent_dir)

def latest_checkpoint(directory):
    checkpoint_dir = None
    checkpoint_num = -1

    dirs = os.listdir(directory)
    print(f'Searching in {directory}: Found {len(dirs)} items')
    for d in dirs:
        full_path = f'{directory}/{d}'
        if os.path.isdir(full_path):
            match = re.match(r'checkpoint-(\d+)', d)
            if match:
                num = int(match.group(1))
                if num > checkpoint_num:
                    checkpoint_num = num
                    checkpoint_dir = full_path

    if checkpoint_dir is None:
        print(f"No directories matching 'checkpoint-<number>' found in '{directory}'.")
        return None, -1
    else:
        print(f"  Latest checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir, checkpoint_num

def load_losses(checkpoint_dir):
    # Lists to store the data
    train_epochs, train_losses, train_steps, eval_epochs, eval_losses, eval_steps = [], [], [], [], [], []

    with open(f'{checkpoint_dir}/trainer_state.json', 'r') as f:
        data = json.load(f)
        for entry in data.get('log_history', []):
            epoch = entry.get('epoch') # Epoch is always expected
            step = entry.get('step') # Same
            if epoch is not None:
                if 'loss' in entry:
                    loss = entry.get('loss')
                    if loss is not None:
                        train_epochs.append(epoch)
                        train_losses.append(loss)
                        train_steps.append(step)
                if 'eval_loss' in entry:
                    eval_loss = entry.get('eval_loss')
                    if eval_loss is not None:
                        eval_epochs.append(epoch)
                        eval_losses.append(eval_loss)
                        eval_steps.append(step)

    return train_epochs, train_losses, train_steps, eval_epochs, eval_losses, eval_steps