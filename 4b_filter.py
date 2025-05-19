import csv
import yaml

"""
Copy CSV from input to output but only if they are in emoji_data.yaml (no skin tones)
"""

INPUT_CSV_PATH =  'emoji/training_data_v2.csv'
YAML_PATH = 'emoji/emoji_data.yaml'
OUTPUT_CSV_PATH = 'emoji/training_data_filtered.csv'

# Load the emoji dictionary from YAML
with open(YAML_PATH, 'r', encoding='utf-8') as f_yaml:
    emoji_dict = yaml.safe_load(f_yaml)

# Open input CSV and output CSV
with open(INPUT_CSV_PATH, 'r', newline='', encoding='utf-8') as infile, \
     open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read and write header
    header = next(reader)
    writer.writerow(header)

    # Process data rows
    for row in reader:
        # Check if row is not empty and the first element is in the dict keys
        if row and row[0] in emoji_dict:
            writer.writerow(row)

print(f"Filtering complete. Output saved to '{OUTPUT_CSV_PATH}'.")
