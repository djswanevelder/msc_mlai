import os
import csv
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# --- Configuration Data ---
# These are the fixed class combinations you provided.
CLASS_COMBINATIONS_DATA: List[Dict[str, Any]] = [
    {'id': 1, 'class1': 'English_setter', 'class2': 'Irish_setter', 'class3': 'Australian_terrier', 'type': 1},
    {'id': 2, 'class1': 'Siberian_husky', 'class2': 'malamute', 'class3': 'Eskimo_dog', 'type': 1},
    {'id': 3, 'class1': 'Persian_cat', 'class2': 'Siamese_cat', 'class3': 'Egyptian_cat', 'type': 1},
    {'id': 4, 'class1': 'Granny_Smith', 'class2': 'orange', 'class3': 'lemon', 'type': 1},
    {'id': 5, 'class1': 'airliner', 'class2': 'warplane', 'class3': 'space_shuttle', 'type': 1},
    {'id': 6, 'class1': 'Siberian_husky', 'class2': 'Grand_piano', 'class3': 'English_setter', 'type': 2},
    {'id': 7, 'class1': 'kit_fox', 'class2': 'umbrella', 'class3': 'grey_whale', 'type': 2},
    {'id': 8, 'class1': 'African_elephant', 'class2': 'guitar', 'class3': 'lab_coat', 'type': 2},
    {'id': 9, 'class1': 'cougar', 'class2': 'baseball', 'class3': 'garden_spider', 'type': 2},
    {'id': 10, 'class1': 'rocking_chair', 'class2': 'strawberry', 'class3': 'ostrich', 'type': 2},
    {'id': 11, 'class1': 'English_setter', 'class2': 'Irish_setter', 'class3': 'desk', 'type': 3},
    {'id': 12, 'class1': 'Egyptian_cat', 'class2': 'Persian_cat', 'class3': 'football_helmet', 'type': 3},
    {'id': 13, 'class1': 'African_elephant', 'class2': 'Indian_elephant', 'class3': 'acoustic_guitar', 'type': 3},
    {'id': 14, 'class1': 'airliner', 'class2': 'warplane', 'class3': 'banana', 'type': 3},
    {'id': 15, 'class1': 'Siberian_husky', 'class2': 'malamute', 'class3': 'hammer', 'type': 3}
]

# --- Class Name Standardization Function ---

def standardize_name(name: str) -> str:
    """Converts a class name to lowercase and removes spaces/underscores."""
    return name.lower().replace('_', '').replace(' ', '')

# --- Script Function ---

def generate_hp_sweep_csv(
    combinations_data: List[Dict[str, Any]],
    output_csv_file: str,
    num_hp_samples_per_combination: int,
    seed: int
) -> None:
    """
    Takes fixed class combinations, standardizes class names, and generates a 
    specified number of unique hyperparameter samples for each combination.

    Args:
        combinations_data: A list of dictionaries, where each dict defines a 
                           fixed class combination.
        output_csv_file: Path to the final output CSV file ('sweep.csv').
        num_hp_samples_per_combination: The number of unique hyperparameter 
                                        sets to generate for each class combination.
        seed: The seed for reproducibility.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    sweep_data: List[dict] = []
    
    total_combinations = len(combinations_data)
    total_rows = total_combinations * num_hp_samples_per_combination
    
    print(f"Found {total_combinations} fixed class combinations.")
    print(f"Generating {num_hp_samples_per_combination} hyperparameter samples for each, for a total of {total_rows} configurations...")
    for combo in combinations_data:
        # Standardize the class names immediately after reading the combination data
        class1_std = standardize_name(combo['class1'])
        class2_std = standardize_name(combo['class2'])
        class3_std = standardize_name(combo['class3'])

        for sample_index in range(num_hp_samples_per_combination):
            
            # Base data uses the standardized class names
            row = {
                'combo_id': combo['id'],
                'class1': class1_std,
                'class2': class2_std,
                'class3': class3_std,
                'combo_type': combo['type'],
            }
            
            # --- Generate Random Hyperparameters for this sample ---
            row['seed'] = np.random.randint(0, 10000)
            row['early_epoch'] = np.random.randint(5, 10)
            row['max_epoch'] = int(np.clip(np.random.normal(loc=55, scale=15, size=1)[0], 10, 100))
            row['optimizer'] = 'Adam' if np.random.random() < 0.5 else 'SGD'
            row['learning_rate'] = round(np.random.lognormal(mean=-3.0, sigma=0.8, size=1)[0], 6) # Centered around 1e-3
            
            # Fixed/Default Parameters
            row['store_weight'] = False
            row['status'] = '-'
            
            sweep_data.append(row)
            
    df = pd.DataFrame(sweep_data)
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully generated {len(df)} configurations and saved to '{output_csv_file}'.")
        print("\nExample of standardized class names in output:")
        print(df.head(1)[['class1', 'class2', 'class3']].to_markdown(index=False))

    except Exception as e:
        print(f"Error writing to file: {e}")

# --- Execution Block ---

if __name__ == "__main__":

    OUTPUT_FILE = os.path.join(os.getcwd(), 'data', 'simple_sweep.csv')

    # Parameters for the generation
    NUM_HP_SAMPLES_PER_COMBO = 60
    RANDOM_SEED = 42
    
    generate_hp_sweep_csv(
        combinations_data=CLASS_COMBINATIONS_DATA,
        output_csv_file=OUTPUT_FILE,
        num_hp_samples_per_combination=NUM_HP_SAMPLES_PER_COMBO,
        seed=RANDOM_SEED
    )