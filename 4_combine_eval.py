import yaml
import csv
import random

yaml_input_file = 'validation_set.yaml'
VALIDATION_DATA = 'validation_data.csv'

processed_rows = []

# Read the YAML file, ensuring UTF-8 encoding for emojis
with open(yaml_input_file, 'r', encoding='utf-8') as f_yaml:
    data = yaml.safe_load(f_yaml)

    # Process each key-value pair from the YAML data
    for emoji_key, text_value in data.items():
        for emoji in emoji_key.split(','):
            # New row [Emoji, Text] for each emoji
            processed_rows.append([emoji, text_value])

print(f"Processed {len(processed_rows)} emoji-text pairs from {yaml_input_file}")

# Shuffle the list of rows randomly
random.shuffle(processed_rows)

# Write the shuffled data to the CSV file
with open(VALIDATION_DATA, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['Emoji', 'Text'])
    writer.writerows(processed_rows)

print(f"Successfully created shuffled CSV file: '{VALIDATION_DATA}' with {len(processed_rows)} data rows.")