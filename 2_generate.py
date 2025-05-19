import re
import random
import os
import sys
import yaml
from datetime import datetime
from tqdm import tqdm

from emoji_ai import ollama

FOLDER = 'emoji'
PREFIX = 'data_v2/'
EMOJI_FILE = f'{FOLDER}/emoji_data.yaml'

OUTPUT_TYPE = 'a' # 'w'

# MODEL = 'llama3.2:1b' # 1.24b
# MODEL = 'llama3.2'    # 3.21b
# MODEL = 'llama3.1'    # 8.03b

# MODEL = 'gemma3:1b'   # 1.00b
# MODEL = 'gemma3'      # 4.30b
# MODEL = 'gemma3:12b'  # 12.2b

# MODEL = 'phi4'        # 14.7b

MODELS = ['llama3.1', 'gemma3', 'qwen2.5'] # 

# Languages (other than english)
LANGUAGES = {
    'llama3.1': ['German', 'French', 'Italian', 'Portuguese', 'Hindi', 'Spanish', 'Thai'],
    'gemma3': ['Chinese', 'French', 'Spanish', 'Portuguese', 'German', 'Italian', 'Russian', 'Japanese', 'Korean', 'Vietnamese', 'Thai', 'Arabic', 'Hindi'],
    'qwen2.5': ['Chinese', 'French', 'Spanish', 'Portuguese', 'German', 'Italian', 'Russian', 'Japanese', 'Korean', 'Vietnamese', 'Thai', 'Arabic'],
}

LANGUAGE_SETTINGS = {
    'mixed': {
        'english_frac': 0.6,
        'style_frac': 0.7
    },
    'english': {
        'english_frac': 1.0,
        'style_frac': 0.7
    },
    'multilingual': {
        'english_frac': 0.0,
        'style_frac': 0.0
    }
}

# MODEL_LABEL = MODEL.replace('.', '-').replace(':', '-')


PROMPTS_TO_GEN = ['keywords', 'paragraph', 'conversation'] # 
# PROMPTS_TO_GEN = ['paragraph'] # 

LANGUAGES_TO_GEN = ['english'] # 'multilingual'

TEMPERATURE = 0.9
DATE = datetime.now().strftime('%Y%m%d_%H%M%S')




##     ##    ###    ##       #### ########     ###    ########  #######  ########   ######  
##     ##   ## ##   ##        ##  ##     ##   ## ##      ##    ##     ## ##     ## ##    ## 
##     ##  ##   ##  ##        ##  ##     ##  ##   ##     ##    ##     ## ##     ## ##       
##     ## ##     ## ##        ##  ##     ## ##     ##    ##    ##     ## ########   ######  
 ##   ##  ######### ##        ##  ##     ## #########    ##    ##     ## ##   ##         ## 
  ## ##   ##     ## ##        ##  ##     ## ##     ##    ##    ##     ## ##    ##  ##    ## 
   ###    ##     ## ######## #### ########  ##     ##    ##     #######  ##     ##  ######  

def remove_quotes(text):
    text = text.strip()
    # Define different quote patterns to match
    quote_patterns = [
        # Double quotes (standard)
        r'"([^"]*)"',
        # Single quotes (standard)
        # r"'([^']*)'",
        # Single quotes (improved to handle contractions)
        r"(?<!\w)'(.*?)'(?!\w)",
        # Curly/smart double quotes
        r'“([^”]*)”',
    ]
    
    # Find all quoted text across all patterns
    all_quoted_text = []
    for pattern in quote_patterns:
        matches = re.findall(pattern, text)
        all_quoted_text.extend(matches)
    
    # Calculate total length of original text
    total_length = len(text)
    
    # Check each quoted string to find the shortest one that meets the threshold
    valid_quotes = []
    for quoted_text in all_quoted_text:
        quoted_length = len(quoted_text)
        ratio = quoted_length / total_length if total_length > 0 else 0
        
        if ratio >= 0.7:
            valid_quotes.append((quoted_text, quoted_length))
    
    # If we found valid quotes, return the shortest one
    if valid_quotes:
        # Sort by length (ascending) and return the shortest
        return min(valid_quotes, key=lambda x: x[1])[0]
    else:
        # If no quoted text is long enough, return the original
        return text

def i_cant(text):
    return (
        text.startswith("I can't")
        or text.startswith("I cannot")
    )


# Response validators
def keyword_validator(text, emoji):
    lines = text.strip().split('\n\n')
    max_c = -1
    
    for l in lines:
        c = l.count(',')
        if c > max_c:
            max_c = c
            text = l
    
    if i_cant(text): return None
    
    # Remove 'Sure. Here's some keywords:'
    if ':' in text:
        text = text.split(':')[1].strip()
    
    if text.endswith('.'):
        text = text[:-1]

    words = []
    for w in text.split(','):
        w = w.replace('_', ' ').replace('-', ' ')
        w.strip().lower()

        # Remove emoji character from start and end of keyword
        w = w[len(emoji):] if w.startswith(emoji) else w
        w = w[:-len(emoji)] if w.endswith(emoji) else w
        
        if w:
            words.append(w)
    
    return ', '.join(words)


def comment_validator(text, emoji):
    # text = text.strip()
    text = remove_quotes(text)

    if text.endswith(emoji):
        text = text[:-len(emoji)].strip()

    # # Remove the "Sure, here's a comment...\n\n" if multiple lines
    # if '\n\n' in text:
    #     text = text.split('\n\n')[1]
    
    if i_cant(text): return None

    return text


def conversation_validator(text, emoji):
    text = text.strip()
    pattern = r'\*\*(.*?)\*\*'

    lines = []
    if '\n\n' in text:
        for chunk in text.split('\n\n'):
            if '\n' in chunk:
                spl = chunk.split('\n')
                name = spl[0].strip()
                message = spl[1].strip()
                if message:
                    lines.append(f'{name} - {remove_quotes(message)}')
    
    if not lines:
        for line in text.split('\n'):
            if ':' in line:
                spl = re.sub(pattern, r'\1', line).split(':')
                name = spl[0].strip()
                if name.startswith('• '):
                    name = name[2:]
                message = spl[1].strip()
                if message:
                    lines.append(f'{name}: {remove_quotes(message)}')

    if not lines:
        return None
    
    return '\n'.join(lines)




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######  
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ## 
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##       
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######  
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ## 
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ## 
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######  

SYSTEM = (
    'You are generating synthetic text data to train LLMs. '
    'Keep responses realistic and representative. '
    'Return only the text without any introduction or explanation.'
)

RELATIONS = [
    'matches',
    'could be represented by',
    'captures the meaning of',
    'feels like',
    'relates to',
# Claude:
    'evokes the essence of',
    'symbolizes',
    'corresponds to',
    'aligns with',
    'is conceptually similar to',
    'conveys the same sentiment as',
    'expresses the emotion behind',
    'reflects the mood of',
    'illustrates the concept of',
    'embodies the spirit of',
    'parallels the meaning of',
    'paints a picture similar to',
    'channels the energy of',
    'invokes the same reaction as',
    'has the same connotation as',
    'represents',
    'brings to mind',
    'serves as an analogy for',
    'connects thematically with',
# GPT 4.5:
    'evokes',
    'conveys the spirit of',
    'is associated with',
    'communicates the feeling of',
    'is inspired by',
    'echoes',
    'suggests',
# Add these to your RELATIONS list (Claude 2nd)
    'creates the atmosphere of',
    'tells a story implied by',
    'conveys the subtext behind',
    'articulates what is implied by',
    'translates the visual language of',
    'captures the cultural meaning of',
    'provides the emotional backdrop for',
    'unpacks the sentiment contained in',
    'gives voice to what is shown in',
    'personifies the character in',
    'translates the non-verbal cue of',
    'transforms into words what is expressed by',
]
STYLES = [
    # Basic intensity levels
    'subtly',
    'noticeably',
    'clearly',
    'strongly',
    'powerfully',
    'overwhelmingly',
    'unmistakably',
    'undeniably',
    'unquestionably',

    # Emotional qualities
    'vividly',
    'profoundly',
    'intensely',
    'lightheartedly',
    'playfully',
    'dramatically',
    'solemnly',
    'enthusiastically',
    'cautiously',
    'boldly',
    'tenderly',

    # Rhetorical styles
    'humorously',
    'sarcastically',
    'ironically',
    'metaphorically',
    'literally',
    'poetically',
    'eloquently',
    'crudely',
    'tactfully',
    'bluntly',
    'explicitly',
    'frankly',

    # Temporal aspects
    'persistently',
    'gradually',
    'suddenly',
    'consistently',
    'frequently',
    'perpetually',

    # Cognitive aspects
    'consciously',
    'intuitively',
    'deliberately',
    'thoughtfully',
    'impulsively',
    'reflectively',
    'mindfully',
    'attentively',
]

SUBCATEGORY_MULT = {
    'base': 3.0,
    'light': 1.0,
    'medium-light': 1.0,
    'medium': 1.0,
    'medium-dark': 1.0,
    'dark': 1.0,
    'male': 2.0,
    'female': 2.0,
    'right': 2.0,
}

PROMPT = {
    'keywords': {
        'min': 1.0,
        'max': 3.0,
        'exp': 1.0,
        'tokens': 200,
        'validator': keyword_validator,
        'verbs': ['Write a'],
        'objects': ['list of keywords'],
        'template': "{verb} comma-separated {object} that {emoji_string}{language}.",
    },
    'paragraph': {
        'min': 1.0,
        'max': 5.0,
        'exp': 1.0,
        'tokens': 400,
        'validator': comment_validator,
        'verbs': ['Write', 'Create'],
        'objects': [
            "a short paragraph",
            "a short sentence",
            "a short social media comment",
            "a short text message",
            "a long paragraph",
            "a long sentence",
            "a long social media comment",
            "a long text message",
            "a reaction to some news",
            "a brief product/movie review",
            "an Instagram caption",
            "a news headline",
            "a tweet (under 280 characters)",
            "a short diary entry",
            "a status update",
            "a catchy slogan or tagline",
            "a short poem",
            "a verse of song lyrics",
            "a thank you message",
            "an apology",
            "a step in a how-to guide",
            "a motivational quote",
            "an invitation to an event",
            "a mobile app notification",
            "a public announcement",
            "a piece of advice",
            "a customer complaint",
            "a product endorsement",
            "a warning or caution",
            "a line from a job posting",
            "a line from a dating profile",
            "a very short story",
        ],
        'template': '{verb} {object} that {emoji_string}{language}, without using the emoji directly.',
    },
    'conversation': {
        'min': 1.0,
        'max': 2.0,
        'exp': 1.0,
        'tokens': 600,
        'validator': conversation_validator,
        'verbs': [
            'Write a short conversation',
            'Create a brief dialogue',
            'Write a quick chat',
            'Create a short interaction',
            'Write a short exchange',
        ],
        'objects': [
            "between two people",
            "between an AI and their user",
            "between friends",
            "between colleagues",
            "between a parent and child",
            "between a customer and service representative",
            "among a group of friends",
            "in a dating app chat",
        ],
        'template': (
            "{verb} {object} that {emoji_string}{language}, without using the emoji directly. "
            "Give only the conversation, with each turn on a new line, prefixed with the participant's name (randomly generated)."
        ),
    },
}

######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######  
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ## 
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##       
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######  
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ## 
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ## 
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######  

def normalize_rank(n, max_n, min_out, max_out, exp):
    if not n:
        return int(min_out)
    if n == 1:
        return int(max_out)
    normalized = (n - 1) / (max_n - 1)    # Normalize to [0, 1]
    transformed = (1 - normalized) ** exp # Flip and add exponential
    final = min_out + transformed * (max_out - min_out) # Scale to [min, max]
    return int(round(final))


def get_subcategories(emojis):
    hexes = [
        ('1F3FB', 'light'),
        ('1F3FC', 'medium-light'),
        ('1F3FD', 'medium'),
        ('1F3FE', 'medium-dark'),
        ('1F3FF', 'dark'),
        ('2642', 'male'),
        ('2640', 'female'),
        ('27A1', 'right'),
    ]
    subcategories = {
        'base': [],
        'light': [],
        'medium-light': [],
        'medium': [],
        'medium-dark': [],
        'dark': [],
        'male': [],
        'female': [],
        'right': [],
    }
    for emoji, data in emojis.items():
        in_base = True
        for h, c in hexes:
            # contains modifier but is not only the modifier
            if h != data['hex'] and h in data['hex']:
                subcategories[c].append(emoji)
                in_base = False

        if in_base: # add to base if added to nothing else
            subcategories['base'].append(emoji)

    return subcategories




########  ########   #######   ######  ########  ######   ######  
##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ## 
##     ## ##     ## ##     ## ##       ##       ##       ##       
########  ########  ##     ## ##       ######    ######   ######  
##        ##   ##   ##     ## ##       ##             ##       ## 
##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ## 
##        ##     ##  #######   ######  ########  ######   ######  

def process_prompt(emojis, prompt_type, model, language):
    if language == 'multilingual':
        languages = LANGUAGES[model]
    else:
        languages = [] # English
    label = model.replace('.', '-').replace(':', '-')
    # If restart == output the file will be loaded and updated in place
    restart_file = f'{FOLDER}/{PREFIX}{language}_{prompt_type}_{label}.yaml' # f''
    output_file = f'{FOLDER}/{PREFIX}{language}_{prompt_type}_{label}.yaml' # _{DATE}
    verbs = PROMPT[prompt_type]['verbs']
    objects = PROMPT[prompt_type]['objects']
    template = PROMPT[prompt_type]['template']
    min_gens = PROMPT[prompt_type]['min'] # Minimum generation count
    max_gens = PROMPT[prompt_type]['max'] # Max generation count (for more popular emojis)
    exponent = PROMPT[prompt_type]['exp'] # Exponent for scaling between min and max
    validator = PROMPT[prompt_type]['validator']
    max_tokens = PROMPT[prompt_type]['tokens']
    english_frac = LANGUAGE_SETTINGS[language]['english_frac']
    style_frac = LANGUAGE_SETTINGS[language]['style_frac']
    
    title = prompt_type.title()

    sub_counts = {
        'base': 0,
        'light': 0,
        'medium-light': 0,
        'medium': 0,
        'medium-dark': 0,
        'dark': 0,
        'male': 0,
        'female': 0,
        'right': 0,
    }

    subcategories = get_subcategories(emojis)

    # Calculate prompts per emoji and total generations
    rank_count = sum(1 for data in emojis.values() if data['rank'])
    total_generations = 0

    max_prompts = 0
    min_prompts = 1_000_000
    
    for emoji, data in emojis.items():
        multiplier = 1.0
        for c in SUBCATEGORY_MULT:
            if emoji in subcategories[c]:
                multiplier = SUBCATEGORY_MULT[c]
                sub_counts[c] += 1
                break

        prompts = normalize_rank(data.get('rank', None), rank_count, min_gens, max_gens*multiplier, exponent)
        
        data['generations'] = prompts
        total_generations += prompts

        if prompts > max_prompts:
            max_prompts = prompts

        if prompts < min_prompts:
            min_prompts = prompts

    emoji_dict = {}
    emoji_keys = {}
    if restart_file and os.path.exists(restart_file):
        with open(restart_file, 'r', encoding='utf8') as f:
            emoji_dict = yaml.safe_load(f)
            if not emoji_dict:
                emoji_dict = {}
    
    if emoji_dict:
        for k in emoji_dict: # emoji_keys['😀'] = ['😀0', '😀1', ...]
            e = k.rstrip('0123456789')
            if e in emoji_keys:
                emoji_keys[e] += [k]
            else:
                emoji_keys[e] = [k]

    # For tracking summary data
    start_time = datetime.now()
    last_time = start_time

    cat_string = ', '.join(f'{n/len(emojis):.1%} {c}' for c, n in sub_counts.items())
    
    # Create the summary header
    start_string = (
        f"# Emojis: {len(emojis)}\n"
        f"# Categories: {cat_string}\n"
        f"# Min: {min_gens}, Max: {max_gens}, Exponent: {exponent}\n"
        f"# Total generations: {total_generations}\n"
        f"# Avg: {total_generations/len(emojis)}, Min: {min_prompts}, Max: {max_prompts}\n"
        f"# Loaded: {len(emoji_dict)}\n"
        f"# Model: {model}, Temp: {TEMPERATURE}\n"
        f"# Start time: {start_time}\n"
        f'# Prompt: {template}\n'
    )
    for n, v in enumerate(verbs, 1):
        start_string += f'# Verb {n}: {v}\n'
    for n, o in enumerate(objects, 1):
        start_string += f'# Object {n}: {o}\n'

    print(start_string)

    # sys.exit() # END BEFORE STARTING (FOR TESTING)

    # Output 'w': overwrite file, output 'a': append to file
    with open(output_file, OUTPUT_TYPE, encoding='utf-8') as f:
        f.write(start_string)
        if OUTPUT_TYPE == 'w' and emoji_dict:
            yaml.dump(emoji_dict, f, allow_unicode=True, width=float("inf"))
    
    # For tracking summary data
    total_tokens = 0
    tps = 0.0
    skipped_emojis = []
    duplicate_emojis = []
    
    progress_bar = tqdm(
        total=total_generations-len(emoji_dict),
        ncols=124,
        desc=f'{title} -  #- (-/-)',
        unit='gen',
        smoothing=0.05,
        postfix='token/s'
    )

    for emoji, data in emojis.items():
        generations = data['generations']
        english_count = int(generations*english_frac)
        available_objects = objects.copy()
        available_languages = languages.copy()
        rank = data['rank']

        for i in range(generations):
            key = f'{emoji}{i}'
            if key in emoji_dict:
                continue

            relation = random.choice(RELATIONS)
            if prompt_type != 'keyword' and random.random() < style_frac:
                style = random.choice(STYLES)
                relation = f"{style} {relation}"

            emoji_string = f"{relation} the emoji {emoji} ({data['name']})"

            # Reset elements if they have been used
            if not available_objects:
                available_objects = objects.copy()
            if available_languages == []:
                available_languages = languages.copy()

            # Choose and remove elements to avoid repetition
            object = random.choice(available_objects)
            available_objects.remove(object)

            if languages and i >= english_count:
                lang = random.choice(available_languages)
                available_languages.remove(lang)
                language = ' in ' + lang
            else: # Default to English
                language = ''

            prompt = template.format(verb=random.choice(verbs), object=object, emoji_string=emoji_string, language=language)

            response, tokens = ollama(prompt, model, SYSTEM, TEMPERATURE, max_tokens)

            generated_text = validator(response, emoji)

            # Print the result
            # objects must start with 'Write/Create a/an...'
            pattern = r'^(?:Write|Create) a(?:n)? (.*?) the emoji '
            match = re.match(pattern, prompt)
            snippet = match.group(1) if match else 'ERROR'
            # snippet = re.sub(pattern, '...', prompt.split(' the emoji ')[0]) # , flags=re.IGNORECASE
            prefix = f"{key} ({data['name']}) - ...{snippet} {emoji}{language}"
            tqdm.write(prefix)

            # Check if this emoji already has an entry with the same value
            skip = False
            if generated_text:
                total_tokens += tokens
                if emoji in emoji_keys:
                    for k in emoji_keys[emoji]:
                        if emoji_dict[k] == generated_text:
                            tqdm.write(f"SKIPPING DUPLICATE ({k}): {generated_text.replace('\n','\n  ')}")
                            skip = True
                            duplicate_emojis += [emoji]
                            break
                    else:
                        tqdm.write(f"  {generated_text.replace('\n','\n  ')}")
                else:
                    tqdm.write(f"  {generated_text.replace('\n','\n  ')}")
            else:
                tqdm.write(f"INCORRECT FORMAT: {response.replace('\n','\n  ')}")
                skip = True
                skipped_emojis += [emoji]
            
            # Update progress bars # f'{title} {emoji}  {failed*100:.1f}% dropped'.ljust(30)+'Done'
            if data['parent']:
                progress_bar.desc = f"{title} {emoji}:{data['parent']}  #{rank if rank else 'N/A'} ({i+1}/{generations})".ljust(30)+' Done'
            else:
                progress_bar.desc = f"{title} {emoji}  #{rank if rank else 'N/A'} ({i+1}/{generations})".ljust(30)+' Done'
            
            now = datetime.now()
            tps = 0.8*tps + 0.2*(tokens / (now - last_time).total_seconds())
            last_time = now
            progress_bar.postfix = f'{tps:.3g}tok/s'
            progress_bar.update(1)
  
            if skip:
                continue

            # Append data to YAML
            for attempt in range(3):
                try:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        yaml.dump({key: generated_text}, f, allow_unicode=True, width=float("inf")) # , default_flow_style=False
                    emoji_dict[key] = generated_text # .strip()
                    if emoji in emoji_keys:
                        emoji_keys[emoji] += [key]
                    else:
                        emoji_keys[emoji] = [key]
                    break
                except Exception as e:
                    tqdm.write(f"Attempt {attempt + 1} failed: {e}")
            else:
                tqdm.write(f"Failed to append {key} data to YAML")
    
    progress_bar.close()
    
    # Calculate summary statistics
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    time_per_gen = total_time / total_generations
    tokens_per_second = total_tokens / total_time
    tokens_per_gen = total_tokens / total_generations
    
    # Write summary data to the YAML file
    summary_data = (
        f"# Generated {total_generations} outputs in {total_time:.2f}s ({time_per_gen:.2f} each)\n"
        f"# Averaged {tokens_per_second:.2f}token/s ({tokens_per_gen:.1f} tokens each generation)\n"
        f"# Skipped ({len(skipped_emojis)}): {' '.join(skipped_emojis)}\n"
        f"# Duplicate ({len(duplicate_emojis)}): {' '.join(duplicate_emojis)}\n"
        f"# End time: {end_time}\n"
        f"################################\n"
    )
    print(summary_data)

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(summary_data)




##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ## 


def main():
    # Load emoji data
    if not os.path.exists(EMOJI_FILE):
        print(f"Error: {EMOJI_FILE} not found. Run emoji.py first.")
        sys.exit(1)
    
    emojis = {}
    with open(EMOJI_FILE, 'r', encoding='utf-8') as f:
        emojis = yaml.safe_load(f)
    
    if not emojis:
        print("No emojis found in the emoji data file.")
        sys.exit(1)

    for model in MODELS:
        for prompt in PROMPTS_TO_GEN:
            for language in LANGUAGES_TO_GEN:
                process_prompt(emojis, prompt, model, language)


# Function to represent multi-line strings as literal blocks
def represent_multiline_str(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# Register the representer
yaml.add_representer(str, represent_multiline_str)


if __name__ == "__main__":
    main()