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
    seed: int
) -> None:
    """
    Reads class names, generates random class permutations, assigns random 
    hyperparameters, and saves the complete configuration to a single CSV file.

    Args:
        map_file_path: Path to the file with class names (e.g., 'imagenet_map.txt').
        output_csv_file: Path to the final output CSV file ('sweep.csv').
        num_permutations: The number of random class combinations to generate (i.e., rows).
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
                    class_names.append(parts[2])
    except Exception as e:
        print(f"Error reading from file: {e}")
        return

    if len(class_names) < 3:
        print("Error: Not enough classes in the file to create permutations of 3.")
        return

    print(f"Found {len(class_names)} classes. Generating {num_permutations} full configurations...")

    sweep_data: List[dict] = []
    
    for i in range(num_permutations):
        sampled_classes = random.sample(class_names, 3)
        sweep_data.append({
            'class1': sampled_classes[0],
            'class2': sampled_classes[1],
            'class3': sampled_classes[2]
        })
    
    df = pd.DataFrame(sweep_data)
    num_rows = len(df)
    

    # Random Hyperparameters
    df['seed'] = np.random.randint(0, 10000, size=num_rows)
    df['early_epoch'] = np.random.randint(15, 25, size=num_rows)
    df['max_epoch'] = np.clip(np.random.normal(loc=100, scale=15, size=num_rows), 50, 150).astype(int)
    df['optimizer'] = np.where(np.random.random(num_rows) < 0.5, 'Adam', 'SGD')
    df['learning_rate'] = np.random.lognormal(mean=-3.0, sigma=0.8, size=num_rows).round(6) # Centered around 1e-3
    
    # Fixed/Default Parameters
    df['store_weight'] = True
    df['status'] = '-'
    
    try:
        df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully generated {num_rows} configurations and saved to '{output_csv_file}'.")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":

    MAP_FILE_PATH = os.path.join(os.getcwd(), 'data', 'imagenet_data', 'imagenet_map.txt') 

    OUTPUT_FILE = os.path.join(os.getcwd(), 'data', 'sweep.csv')



    generate_full_sweep_csv(
        map_file_path=MAP_FILE_PATH,
        output_csv_file=OUTPUT_FILE,
        num_permutations=100, 
        seed=42
    )