import os
import yaml
from datetime import datetime

INPUT_FOLDER = 'input'
input_files = [
    f'{INPUT_FOLDER}/all_emoji.tsv',
    # f'{INPUT_FOLDER}/mod_emoji.tsv',
    # f'{INPUT_FOLDER}/zwj_emoji.tsv'
]
ranking_file = f'{INPUT_FOLDER}/2021_ranked.tsv'
EMOJI_FILE = f'emoji_data.yaml'

def extract_emojis(input_paths, output_path=None, ranking_path=None):
    """Extract emojis from the input files"""
    emojis = {}
    rank_count = 0
    parent_count = 0
    cat = ''
    sub = ''
    num = 0
    for file_num, path in enumerate(input_paths):
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header line
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t')
                    k = None

                    if len(parts) == 6: # zwj
                        i, u, _, e, _, n = parts
                    elif len(parts) == 5: # all
                        i, u, e, n, k = parts
                    elif len(parts) == 4: # mod
                        i, u, e, n = parts
                    else:
                        print('ERROR')
                        continue

                    if e in emojis:
                        # print(f'DEBUG: Emoji {e} aready exists, path={file_num}')
                        continue

                    i = num + int(i)
                    u = u.replace('U+', '')

                    emojis[e] = {
                        'num': i,
                        'hex': u,
                        'name': n,
                        'category': cat,
                        'subcategory': sub,
                        'keywords': k,
                        'parent': None,
                        'rank': None
                    }

                    # Find possible parents and update connection...
                    if file_num == 0 or ':' not in n:
                        continue

                    pre, suf = n.split(': ')

                    if pre in ('kiss', 'couple with heart'):
                        people = ('woman', 'man')
                        spl = suf.split(', ')
                        if (spl[0] in people and spl[1] in people):
                            pre = f'{pre}: {spl[0]}, {spl[1]}'

                    for ee in emojis: # parent loop
                        if emojis[ee]['name'] == pre:
                            emojis[e]['parent'] = ee
                            parent_count += 1
                            break # parent loop
                else:
                    if line[0].isupper():
                        cat = line
                    else:
                        sub = line
        num = i

        print(f'File: {path}')
        print(f'Emoji count: {len(emojis)}')
        # parent_count = sum(1 for data in emojis.values() if data['parent'])
        print(f'With parents: {parent_count}')

    # Add rankings from rankings file...

    if not ranking_path or not os.path.exists(ranking_path):
        return emojis

    with open(ranking_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header line
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 7: # rank
                _, r, e, *_ = parts
                r = int(r)

                if r > 0:
                    # Try removing VS16 if not in dictionary
                    if e not in emojis:
                        emoji = e.replace('\uFE0F', '')
                    else:
                        emoji = e
                        
                    if emoji in emojis:
                        emojis[emoji]['rank'] = r
                        rank_count += 1
                    else:
                        print(f'ERROR: Got rank {r} for nonexistant emoji {e}')
            else:
                print(f'ERROR: Invalid format: {parts}')
    
    # Output YAML file...

    if not output_path:
        return emojis

    # Create the summary header
    start_string = (
        f"# Emojis: {len(emojis)}\n"
        f"# Ranked: {rank_count}\n"
        f"# Parents: {parent_count}\n"
        f"# Date: {datetime.now()}\n"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(start_string)
        yaml.dump(emojis, f, allow_unicode=True, width=float("inf"), sort_keys=False)

    return emojis

def main():
    emojis = extract_emojis(input_files, EMOJI_FILE, ranking_file)
    print(f"Extracted {len(emojis)} emojis to {EMOJI_FILE}")

if __name__ == "__main__":
    main()