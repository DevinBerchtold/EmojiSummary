import yaml
import csv
import os
import time
from tqdm import tqdm
import random

from emoji_ai import ollama

EMOJI_FILE = f'emoji/emoji_data.yaml'

FOLDER = 'emoji/data_v2'
MODELS = ['llama3.1', 'gemma3', 'qwen2.5']
SEED = 1337

# MODEL = 'llama3.2:1b' # 1.24b
# MODEL = 'llama3.2'    # 3.21b
MODEL = 'llama3.1'    # 8.03b

# MODEL = 'gemma3:1b'   # 1.00b
# MODEL = 'gemma3'      # 4.30b
# MODEL = 'gemma3:12b'  # 12.2b

# MODEL = 'phi4'        # 14.7b
# MODEL_LABEL = MODEL.replace('.', '-').replace(':', '-')

# GRADER = 'llama3.1'
# GRADER = 'gemma3'
# GRADER_LABEL = GRADER.replace('.', '-').replace(':', '-')

TEMPERATURE = 0.1

LANGUAGES_TYPES = ['english', 'multilingual']
FILES_TYPES = ['keywords', 'paragraph', 'conversation']

FRACTION = 1.0 # Low number can be used for test runs

PREFIX = ''

SYSTEM = 'Answer the following yes/no question.'

def check_emoji_text_match(emoji, text, model='llama3', max_retries=5):
    prompt = f"""Does the following emoji generally match the text?

Emoji: {emoji}
Text: {text}

Answer with only "yes" if it matches or "no" if it doesn't match."""
    
    for attempt in range(max_retries):
        try:
            answer = ollama(prompt, model, SYSTEM, TEMPERATURE, 20)[0].lower()

            if 'yes' in answer:
                return True
            elif 'no' in answer:
                return False
            else:
                return True # Unclear response, keep by default

        except Exception as e:
            print(f"Error {emoji} on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(1) # Wait before retrying
    
    print(f"Could not complete API call for {emoji} after {max_retries} attempts. Keeping by default.")
    return True

def main():
    # Load emoji data
    emojis = {}
    with open(EMOJI_FILE, 'r', encoding='utf-8') as f:
        emojis = yaml.safe_load(f)

    random.seed(SEED)
    for grader in MODELS:
        grader_label = grader.replace('.', '-').replace(':', '-')
        for model in MODELS:
            model_label = model.replace('.', '-').replace(':', '-')
            for l in LANGUAGES_TYPES:
                for t in FILES_TYPES:
                    print(f'Model: {model}, Grader: {grader}')
                    filename = f'{FOLDER}/{PREFIX}{l}_{t}_{model_label}'
                    title = t.title()
                    # Load keywords
                    infile = f'{filename}.yaml'
                    with open(infile, 'r', encoding='utf-8') as file:
                        all_emojis = yaml.safe_load(file) or {}
                    print(f"Loaded {len(all_emojis)} entries from {infile}")

                    # Validate emoji-keyword matches
                    valid_emojis = []

                    # Keep only a fraction
                    num_to_remove = len(all_emojis) - int(len(all_emojis) * FRACTION)
                    if num_to_remove != 0:
                        keys_to_remove = random.sample(list(all_emojis.keys()), num_to_remove)
                        for key in keys_to_remove:
                            del all_emojis[key]

                    total = len(all_emojis)

                    # Setup tqdm progress bar
                    progress_bar = tqdm(
                        total=total,
                        ncols=120,
                        desc=f'{title} - -% dropped',
                        unit='gens',
                        smoothing=0.05,
                    )

                    for i, (key, text) in enumerate(all_emojis.items(), 1):
                        # Remove the number suffix from the emoji for the prompt
                        emoji = key.rstrip('0123456789')
                        name = emojis[emoji]['name']
                        emoji_str = f'{emoji} ({name})' if name else emoji
                        if emoji in text:
                            tqdm.write(f"({i}) EMOJI IN TEXT - {emoji_str} - {text}")
                        else:
                            if check_emoji_text_match(emoji_str, text, model=grader):
                                valid_emojis.append((emoji, text))
                            else:
                                tqdm.write(f"({i}) THEME MISMATCH - {emoji_str} - {text}")

                        # Update progress bar description
                        failed = (i-len(valid_emojis))/i
                        progress_bar.desc = f'{title} {emoji}  {failed*100:.1f}% dropped'.ljust(30)+'Done'
                        progress_bar.update(1)

                    progress_bar.close()

                    # Save cleaned keywords to CSV
                    outfile = f'{filename}_{grader_label}.csv'
                    with open(outfile, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows([['Emoji', 'Text']] + valid_emojis)
                    tqdm.write(f'Saved cleaned entries to {outfile}')
                    tqdm.write(f"Finished: Kept {len(valid_emojis)} entries, removed {len(all_emojis) - len(valid_emojis)} entries.")

if __name__ == "__main__":
    main()