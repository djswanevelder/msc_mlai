import csv
import io
import sys
from typing import List, Dict, Any, Tuple

def generate_artifact_metadata_csv(input_data_content: str) -> str:
    """
    Parses CSV data containing multiple runs, expands each run into multiple 
    rows based on the 'artifacts' field, and generates a new CSV string.

    Args:
        input_data_content: The entire content (as a string) of the input CSV file.

    Returns:
        A string containing the new CSV data with one row per artifact.
    """
    # Use StringIO to treat the input string as a file
    input_file = io.StringIO(input_data_content)
    
    # Read the input CSV
    reader = csv.DictReader(input_file)
    
    # List to collect all processed artifact rows
    all_artifact_rows: List[Dict[str, Any]] = []

    # Iterate over every run row in the input CSV
    for run_data in reader:
        # --- 1. Extract required data fields ---
        artifacts_string = run_data.get('artifacts', '')
        # Include run_id to link back to the original source
        # run_id = run_data.get('run_id', 'N/A')
        val_loss = run_data.get('val_loss', 'N/A')
        class1 = run_data.get('class1', 'N/A')
        class2 = run_data.get('class2', 'N/A')
        class3 = run_data.get('class3', 'N/A')

        artifact_ids: List[str] = [
            # Strip whitespace, then use rsplit(':', 1) to split only on the last colon.
            # We take the first element ([0]), which is the ID without the ":vX" suffix.
            art.strip().rsplit(':', 1)[0] 
            for art in artifacts_string.split(',') 
            if art.strip() and "history" not in art # Filter out empty and history artifacts
        ]

        # --- 3. Generate new rows for each artifact ---
        for artifact_id in artifact_ids:
            # We assume the weights (and thus the loss/classes) correspond to the 
            # final state of the run, so we duplicate the metadata for each artifact.
            all_artifact_rows.append({
                'artifact_id': f'{artifact_id}.pth',
                'val_loss': val_loss,
                'class1': class1,
                'class2': class2,
                'class3': class3,
            })

    # --- 4. Prepare and Write Output CSV ---
    output_file = io.StringIO()
    # Added 'run_id' to the output fields for traceability
    fieldnames = ['artifact_id', 'val_loss', 'class1', 'class2', 'class3']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(all_artifact_rows)

    return output_file.getvalue()

def main():
    """
    Main function to handle file path input from command-line arguments and 
    write the resulting metadata to an output file.
    """
    if len(sys.argv) < 3:
        print("Error: Please provide the paths for the input and output CSV files.")
        print("Usage: python process_artifacts.py <path_to_input_file.csv> <path_to_output_file.csv>")
        sys.exit(1)

    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    
    print(f"Reading data from input file: '{input_filepath}'")
    
    # Read the file content
    try:
        with open(input_filepath, 'r') as f:
            input_data = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        sys.exit(1)
    
    # Process the data
    output_csv = generate_artifact_metadata_csv(input_data)
    
    # Write the output to a new file
    try:
        # Use 'w' mode to create/overwrite the file. newline='' is important for CSV files.
        with open(output_filepath, 'w', newline='') as f:
            f.write(output_csv)
        print(f"\nSuccessfully wrote artifact metadata to: '{output_filepath}'")
    except Exception as e:
        print(f"\nError: Could not write to output file '{output_filepath}': {e}")
        sys.exit(1)

# if __name__ == "__main__":
#     # main()


import torch
import numpy as np
import sys
import os
from typing import Dict, Any

def inspect_dataset(file_path: str):
    """
    Loads the aggregated meta-dataset and prints its structure, shapes, and types.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at '{file_path}'.")
        print("Please ensure the file path is correct or run the aggregation script first.")
        sys.exit(1)

    print(f"Attempting to load data from: {file_path}")
    
    # --- FIX: Set weights_only=False to allow loading of mixed content (Tensors and NumPy arrays) ---
    try:
        # We must set weights_only=False to load the NumPy arrays saved alongside the Tensors
        data: Dict[str, Any] = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"\nFATAL ERROR during torch.load:")
        print(f"Could not load the file even with weights_only=False. Check if the file is corrupted.")
        print(f"Original PyTorch error: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary but loaded object type is {type(data)}.")
        return

    print("--- Successfully Loaded Dataset ---")
    
    # Use the length of the weights tensor as the definitive N
    N = len(data.get('weights', [])) if isinstance(data.get('weights'), torch.Tensor) else 'N/A'

    print(f"Total number of runs (N): {N}")
    print("-" * 35)
    print(f"Loaded Dataset Keys: {list(data.keys())}")
    print("-" * 35)

    keys = ['weights', 'dataset_vector', 'final_loss']

    for key in keys:
        if key in data:
            item = data[key]
            print(f"--- {key}:")
            print(f"  Type: {type(item)}")
            
            # Print shape based on type
            if isinstance(item, torch.Tensor):
                print(f"  Shape: {tuple(item.shape)}")
                print(f"  DType: {item.dtype}")
            elif isinstance(item, np.ndarray):
                print(f"  Shape: {item.shape}")
                print(f"  DType: {item.dtype}")
            else:
                print(f"  Content Type: {type(item)}")
        else:
            print(f"--- {key}: NOT FOUND IN DATASET")
    print("-" * 35)


import pandas as pd

if __name__ == "__main__":
    # Default to 'final_meta_dataset.pt' if no argument is provided
    # file_name = sys.argv[1] if len(sys.argv) > 1 else 'final_meta_dataset.pt'
    # inspect_dataset(file_name)
    data = torch.load('encoded-weight-latents.pt')
    model_latents = data['encoded_models']
    
    
    try:
        artifact_df = pd.read_csv('meta-dataset_info.csv')
        artifact_names_set = set(artifact_df['artifact_name'].astype(str))
        print(f"\nLoaded {len(artifact_names_set)} unique artifact names from weights.csv.")
    except FileNotFoundError:
        print("\nError: weights.csv not found. Aborting key search.")
        artifact_names_set = set()
    
    keys_found_in_csv = 0

    print("\n--- Model Latents Check ---")
    
    for key in model_latents:
        latent = model_latents[key]
        
        # 2. Check if the current model latent key exists in the artifact CSV data
        if key in artifact_names_set:
            keys_found_in_csv += 1
            is_present_text = "(FOUND in CSV)"
        else:
            is_present_text = "(NOT found in CSV)"

        print(f'Key: {key} {is_present_text}')
        
        # Original code print, kept for context but modified to break after 5 checks for testing
        # print(latent) 
        if keys_found_in_csv < 5:
            # Simple placeholder logic to show key processing
            pass
        else:
            # 3. Print final count after the loop (or break)
            print(f"\n--- Result ---")
            print(f"Total keys checked: {len(model_latents)}")
            print(f"Total keys found in 'weights.csv': {keys_found_in_csv}")
            # break # Uncomment this if you only want to process the first few keys
    
    # If the loop finished without breaking (i.e., less than 5 keys or no break), print the final count
    if keys_found_in_csv <= len(model_latents):
        print(f"\n--- Final Result ---")
        print(f"Total keys checked: {len(model_latents)}")
        print(f"Total keys found in 'weights.csv': {keys_found_in_csv}")