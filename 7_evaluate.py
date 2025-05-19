import yaml
import os
import time

from emoji_ai import ollama, latest_checkpoint

INPUT_FILE = 'emoji/validation_set.yaml'

def extract_checkpoint_number(checkpoint_name):
    try:
        return int(checkpoint_name.split('-')[1])
    except (IndexError, ValueError):
        return float('inf')

BASE_PATH = 'C:/AI/Emogist/'
# DIRECTORY = f'C:/AI/Emogist/outputs/'
DIRECTORY = ''
TEMPERATURE = 0.5

# TYPE = 'unsloth'
# MODELS = sorted(os.listdir(DIRECTORY), key=extract_checkpoint_number)

MODELS = [
    # ('gemini', 'tunedModels/emoji-v4-vq2xc0e6aoly'),
    # ('gemini', 'tunedModels/emoji-v7-aqr219im8f8'),
    # ('unsloth', 'C:/AI/Emogist/emogist_llama3.2-1b_v0/checkpoints'),
]

EXPERIMENTS = [ # Name, model, version  f'{BASE_PATH}{name}_{model}_v{version}/checkpoints'
    # ('emogist', 'llama3.2-1b', 0, 9000),# 
    # ('emogist', 'llama3.2-1b', 1, 11500),
    # ('emogist', 'llama3.2-1b', 2, 5500),# 
    # ('emogist', 'llama3.2-1b', 3, 5500),
    # ('emogist', 'llama3.2-1b', 4, 9500),# 
    # ('emogist', 'llama3.2-1b', 5, 4500),
    # ('emogist', 'llama3.2-1b', 6, 4500),
    ('emogist', 'llama3.2-3b', 7, 3000),
    ('emogist', 'llama3.2-3b', 8, 2500),
    ('emogist', 'llama3.2-3b', 9, 9000),
    ('emogist', 'llama3.2-3b', 10, 5500),
]

MODELS += [('unsloth', f"{BASE_PATH}{n}_{m}_v{v}/checkpoints/checkpoint-{c}") for n, m, v, c in EXPERIMENTS]

# UNSLOTH_DIRECTORIES = [
#     'C:/AI/Emogist/emogist_llama3.2-1b_v0',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v1',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v2',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v3',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v4',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v5',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v6',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v7',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v8',
#     'C:/AI/Emogist/emogist_llama3.2-1b_v9',
# ]

# MODELS += [('unsloth', latest_checkpoint(f'{d}/checkpoints')[0]) for d in UNSLOTH_DIRECTORIES]

if DIRECTORY:
    dm = sorted(os.listdir(DIRECTORY), key=extract_checkpoint_number)
    MODELS += [('unsloth', DIRECTORY+m) for m in dm]

def unsloth_generate(input, model, tokenizer):
    # messages = [{
    #     "role": "user",
    #     "content": f'Below is some text. Write the emoji which best fits the text\n\n### Text:\n{text.strip()}'
    # }]
    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt = True,
    #     return_tensors = "pt",
    #     # return_attention_mask=True
    # ).to("cuda")

    # # input_ids = inputs['input_ids']
    # # attention_mask = inputs['attention_mask']

    # # Instead of using a streamer for result collection, decode the output
    # outputs = model.generate(
    #     input_ids,            # Pass input_ids explicitly
    #     # attention_mask=attention_mask,  # <-- Pass the attention mask here
    #     max_new_tokens=32,
    #     # pad_token_id=tokenizer.eos_token_id,
    #     # Remove streamer if you don't need real-time output
    # )
        
    # # Decode the generated text (excluding input tokens)
    # return tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    messages = [
        {'role': 'system', 'content': 'Below is some text. Write the emoji which best fits the text.'},
        {'role': 'user', 'content': text.strip()}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True,
    ).to("cuda")

    # --- Generate Response ---
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], # Pass the generated mask
        max_new_tokens=20,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id, # Avoid pad_token warning
        # New Stuff
        # Optional: Add sampling parameters if needed
        # do_sample=True, num_beams = 4, 
        repetition_penalty = 1.2,
        temperature=0.2, # top_p=0.1, 
    )

    # Decode only the newly generated tokens
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)    

def gemini(input, model):
    gen = google_client.models.generate_content(
        model=model,
        contents=input, 
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=10,
            # system_instruction=SYSTEM
        )
    )
    return gen.text or ''

if __name__ == "__main__":
    # Dictionary to store results
    results = {m: [] for (_, m) in MODELS}
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        validation = yaml.safe_load(file)

    validation_list = [(k, v) for k, v in validation.items()]

    for (t, m) in MODELS:
        print(f'\n--- Processing Model: {t} - {m} ---')
        if t == 'unsloth':
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = m,
                max_seq_length = 1024,
                dtype = None,
                load_in_4bit = True,
            )
            FastLanguageModel.for_inference(model) # Enable native 2x faster inference

        elif t == 'gemini':
            from google import genai
            from google.genai import types
            google_client = genai.Client(api_key="AIzaSyA26k_P-RN1r0G1LPYtIy7z8i4VVFQ9_xI")
            model = m

        elif t == 'ollama':
            # from ollama import Client
            # ollama_client = Client()
            from emoji_ai import ollama
            model = m

        print(f'Loading {t}: {m}')
        for i, (emojis, text) in enumerate(validation_list):
            print(f'{emojis} {i}:'.ljust(15), end='')
            # messages = [
            #     {"role": "user", "content": text},
            # ]
            
            if t == 'unsloth':
                generated_text = unsloth_generate(text, model, tokenizer)
            elif t == 'gemini':
                generated_text = gemini(text, model)
            elif t == 'ollama':
                generated_text, _ = ollama(text, model)
            
            print(generated_text)
            results[m].append(generated_text.strip())

        if t == 'unsloth':
            del model
            del tokenizer
            time.sleep(5)
        elif t == 'gemini':
            pass
        elif t == 'ollama':
            pass

    # Save results to YAML file
    with open('results.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(results, file, allow_unicode=True, sort_keys=False)
    
    print(f"Results saved to results.yaml")