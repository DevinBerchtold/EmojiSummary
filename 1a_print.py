import yaml

def print_emojis_by_category():
    # Load emoji data from YAML file
    with open('emoji/emoji_data.yaml', 'r', encoding='utf-8') as file:
        emoji_data = yaml.safe_load(file)
    
    # Group emojis by category
    categories = {}
    for emoji, data in emoji_data.items():
        if isinstance(data, dict) and 'category' in data:
            category = data['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(emoji)
    
    # Print emojis by category, 32 per line
    for category, emojis in categories.items():
        print(f"\n{category}:")
        for i in range(0, len(emojis), 32):
            print(''.join(emojis[i:i+32]))

if __name__ == "__main__":
    print_emojis_by_category()