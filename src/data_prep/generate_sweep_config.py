import os
import csv
import random
import numpy as np
import pandas as pd
from typing import List

def generate_full_sweep_csv(
    map_file_path: str, 
    output_csv_file: str, 
    num_permutations: int, 
    subset_size: int,
    rows_to_mark_done: int,
    seed: int
) -> None:
    """
    Reads class names, selects a subset, generates random class permutations 
    from the subset, assigns random hyperparameters, and saves the configuration to a CSV.

    Args:
        map_file_path: Path to the file with class names (e.g., 'imagenet_map.txt').
        output_csv_file: Path to the final output CSV file ('sweep.csv').
        num_permutations: The number of random class combinations to generate (i.e., rows).
        subset_size: The number of classes to randomly select for the subset.
        seed: The seed for reproducibility.
    """
    if not os.path.exists(map_file_path):
        print(f"Error: The input map file '{map_file_path}' was not found.")
        return
    
    random.seed(seed)
    np.random.seed(seed)

    class_names: List[str] = []
    try:
        with open(map_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                # Assuming the third part is the class name
                if len(parts) >= 3:
                    class_names.append(parts[2].lower())
    except Exception as e:
        print(f"Error reading from file: {e}")
        return

    # --- MODIFICATION START ---
    
    # 1. Check if the requested subset size is valid
    if subset_size < 3:
        print("Error: Subset size must be at least 3 to create permutations of 3.")
        return
    if len(class_names) < subset_size:
        print(f"Error: Requested subset size ({subset_size}) is larger than the total number of classes found ({len(class_names)}).")
        return
        
    # 2. Select the random subset of classes
    # Use the same seed for reproducibility in subset selection
    class_subset = random.sample(class_names, subset_size)
    print(f"Found {len(class_names)} total classes. Selecting a random subset of {subset_size} classes for the sweep.")
    
    # The permutations will be drawn from this subset
    source_classes = class_subset
    
    # --- MODIFICATION END ---

    if len(source_classes) < 3: # Redundant check but good practice
        print("Error: Not enough classes in the source list to create permutations of 3.")
        return

    print(f"Generating {num_permutations} configurations from the {len(source_classes)} selected classes...")

    sweep_data: List[dict] = []
    
    for i in range(num_permutations):
        # The random.sample is now drawing from the smaller 'source_classes' list
        sampled_classes = random.sample(source_classes, 3) 
        sweep_data.append({
            'class1': sampled_classes[0],
            'class2': sampled_classes[1],
            'class3': sampled_classes[2]
        })
    
    df = pd.DataFrame(sweep_data)
    num_rows = len(df)
    
    # Random Hyperparameters (remaining unchanged)
    df['seed'] = np.random.randint(0, 10000, size=num_rows)
    df['early_epoch'] = np.random.randint(10, 20, size=num_rows)
    df['max_epoch'] = np.clip(np.random.normal(loc=50, scale=15, size=num_rows), 20, 80).astype(int)
    df['optimizer'] = np.where(np.random.random(num_rows) < 0.5, 'Adam', 'SGD')
    df['learning_rate'] = np.random.lognormal(mean=-3.0, sigma=0.8, size=num_rows).round(6) # Centered around 1e-3
    
    # Fixed/Default Parameters
    df['store_weight'] = True
    
    mark_done_count = min(rows_to_mark_done, num_rows)
    df['status'] = ['Done'] * mark_done_count + ['-'] * (num_rows - mark_done_count)

    try:
        df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully generated {num_rows} configurations and saved to '{output_csv_file}'.")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":

    MAP_FILE_PATH = os.path.join(os.getcwd(), 'data', 'imagenet_data', 'imagenet_map.txt') 

    OUTPUT_FILE = os.path.join(os.getcwd(), 'data', 'sweep.csv')

    # Define the input parameters for the script
    NUMBER_OF_PERMUTATIONS = 500
    SUBSET_SIZE = 50
    ROWS_TO_MARK_DONE = 0
    RANDOM_SEED = 42

    generate_full_sweep_csv(
        map_file_path=MAP_FILE_PATH,
        output_csv_file=OUTPUT_FILE,
        num_permutations=NUMBER_OF_PERMUTATIONS, 
        subset_size=SUBSET_SIZE,
        seed=RANDOM_SEED,
        rows_to_mark_done=ROWS_TO_MARK_DONE
    )