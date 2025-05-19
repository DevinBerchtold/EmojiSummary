import csv
import glob
import os
import random
from collections import Counter # Import Counter

FOLDER = 'data_v2'
TRAINING_DATA = 'training_data.csv'

def combine_filter_shuffle_csv_files(input_folder, output_file):
    """
    Combines rows from multiple CSV files in a folder, keeping only rows
    that appear at least twice across all files, removes duplicates,
    shuffles the result, and writes it to an output file.
    """
    # Get a list of all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV files to potentially combine.")

    # List to store all rows (excluding headers) from all files
    all_rows_raw = []
    header = None
    files_processed = 0

    # Read each CSV file and collect all rows
    for file in csv_files:
        try:
            with open(file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)

                current_header = next(reader)
                # Get header from the first valid file processed
                if header is None:
                    header = current_header
                    print(f"Using header from: {file}")

                # Add all data rows from this file
                file_rows = list(reader)
                # --- Convert rows to tuples for hashability needed by Counter ---
                all_rows_raw.extend([tuple(row) for row in file_rows])
                print(f"Read {len(file_rows)} rows from {os.path.basename(file)}")
                files_processed += 1

        except Exception as e:
            print(f"Error opening or reading {file}: {e}")

    if not all_rows_raw:
        print("No data could be loaded from the CSV files.")
        return

    print(f"\nRead a total of {len(all_rows_raw)} rows (including duplicates) from {files_processed} files.")

    # Count occurrences of each unique row
    row_counts = Counter(all_rows_raw)
    print(f"Found {len(row_counts)} unique rows across all files.")

    # Filter rows: Keep only those that appeared at least twice
    filtered_unique_rows = [list(row_tuple) for row_tuple, count in row_counts.items() if count >= 2]

    if not filtered_unique_rows:
        print("No rows appeared at least twice across the input files.")
        return
    print(f"Found {len(filtered_unique_rows)} unique rows that appeared at least twice.")

    # Shuffle the selected unique rows
    random.shuffle(filtered_unique_rows)

    # Write to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header) # Write the header
        writer.writerows(filtered_unique_rows) # Write the filtered & shuffled rows

if __name__ == "__main__":
    combine_filter_shuffle_csv_files(FOLDER, TRAINING_DATA)