import yaml
from collections import defaultdict

def load_emoji_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def group_emojis_with_parents(emoji_data):
    grouped = defaultdict(lambda: defaultdict(list))
    # First pass: collect parents
    parent_emojis = {emoji: data for emoji, data in emoji_data.items() if data['parent'] is None}
    # Second pass: add children to parents
    for emoji, data in emoji_data.items():
        category, subcategory = data['category'], data['subcategory']
        if data['parent'] is None:
            grouped[(category, subcategory)][emoji].append(emoji)
        else:
            grouped[(category, subcategory)][data['parent']].append(emoji)

    return grouped

def print_grouped_emojis(grouped, max_emojis_per_line=18):
    for (category, subcat), parent_groups in grouped.items():
        print(f"\n{category}: {subcat}")
        line_count = 0
        line = ""
        for parent, emojis in parent_groups.items():
            emoji_group = ''.join(emojis)
            # Don't break emoji groups across lines
            if line_count + len(emojis) > max_emojis_per_line:
                print(line.strip())
                line = ""
                line_count = 0
            line += emoji_group + ' '
            line_count += len(emojis)
        if line:
            print(line.strip())

# Load emoji data from YAML
emoji_data = load_emoji_data('emoji/emoji_data.yaml')

# Group emojis based on category, subcategory and parents
grouped_emojis = group_emojis_with_parents(emoji_data)

# Print emojis compactly with no more than 20 per line, not breaking parents groups across lines
print_grouped_emojis(grouped_emojis, max_emojis_per_line=20)