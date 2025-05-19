from unsloth import FastLanguageModel
from transformers import TextStreamer

BASE_PATH = 'C:/AI/Emogist/'
NAME = 'emogist_llama3.2-3b_v10'
MODEL = BASE_PATH+NAME
# MODEL = f'C:/AI/Emogist/outputs/checkpoint-17000'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

while True:
    input_text = input('Enter text: ')
    messages = [
        {"role": "user", "content": input_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
