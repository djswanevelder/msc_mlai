import os
import random
import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Any, Tuple

# --- Configuration Data ---
# These are the fixed class combinations provided by the user.
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

# Defines the 6 possible permutations for the three class slots
CLASS_SLOT_PERMUTATIONS = list(itertools.permutations(['class1', 'class2', 'class3']))
NUM_PERMUTATIONS = len(CLASS_SLOT_PERMUTATIONS) # This will be 6

# --- Class Name Standardization Function ---

def standardize_name(name: str) -> str:
    """Converts a class name to lowercase and removes spaces, keeping underscores."""
    # Only removes spaces and converts to lowercase, preserving underscores
    return name.lower().replace(' ', '')

# --- Script Function ---

def generate_hp_sweep_csv(
    combinations_data: List[Dict[str, Any]],
    output_csv_file: str,
    num_hp_samples_per_combination: int,
    seed: int
) -> None:
    """
    Generates a full sweep CSV by:
    1. Taking fixed class combinations.
    2. Generating all 6 permutations of class order for each combination.
    3. Applying a proportional number of unique hyperparameter sets
       to each class order, ensuring a balanced sweep.

    Args:
        combinations_data: A list of dictionaries defining the fixed class combinations.
        output_csv_file: Path to the final output CSV file.
        num_hp_samples_per_combination: The total number of unique hyperparameter 
                                        sets to generate PER fixed class combination. 
                                        MUST be a multiple of 6.
        seed: The seed for reproducibility.
    """
    
    # 1. Setup and Input Validation
    random.seed(seed)
    np.random.seed(seed)
    
    if num_hp_samples_per_combination % NUM_PERMUTATIONS != 0:
        raise ValueError(
            f"Error: 'num_hp_samples_per_combination' ({num_hp_samples_per_combination}) "
            f"must be a multiple of {NUM_PERMUTATIONS} to ensure equal class order distribution."
        )

    sweep_data: List[dict] = []
    
    total_combinations = len(combinations_data)
    num_hp_sets_per_class_order = num_hp_samples_per_combination // NUM_PERMUTATIONS
    total_rows = total_combinations * num_hp_samples_per_combination
    
    print(f"Found {total_combinations} fixed class combinations.")
    print(f"Total permutations per combination: {NUM_PERMUTATIONS}. Each will receive {num_hp_sets_per_class_order} unique HP sets.")
    print(f"Generating a total of {total_rows} configurations...")
    
    # Initialize the sequential row counter
    row_id_counter = 1 
    
    # 2. Main Generation Loop
    for combo in combinations_data:
        # Standardize the original names once
        original_names = {
            'class1': standardize_name(combo['class1']),
            'class2': standardize_name(combo['class2']),
            'class3': standardize_name(combo['class3']),
        }

        # Store the original ID from the source list for traceability
        source_id = combo['id']

        # Iterate over the 6 possible class orderings (Permutations)
        for class_order in CLASS_SLOT_PERMUTATIONS:
            
            # The order maps the column name (e.g., 'class1' slot) to the original class key (e.g., 'class3' name)
            current_order_names = {
                'class1': original_names[class_order[0]],
                'class2': original_names[class_order[1]],
                'class3': original_names[class_order[2]],
            }

            # Generate the required number of unique HP sets for this specific class order
            for _ in range(num_hp_sets_per_class_order):
                
                # Base data row for this combination and class order
                row = {
                    'combo_id': row_id_counter,           # Sequential ID for every row
                    'source_combo_id': source_id,         # Keeps the original ID for grouping
                    'combo_type': combo['type'],
                    # Assigned classes based on the current permutation
                    **current_order_names,
                }
                
                # Increment the counter immediately after assigning the ID
                row_id_counter += 1
                
                # --- Generate Random Hyperparameters ---
                row['seed'] = np.random.randint(0, 10000)
                row['early_epoch'] = np.random.randint(5, 10)
                
                # Max Epoch (Normal distribution, clipped 10 to 100)
                row['max_epoch'] = int(np.clip(np.random.normal(loc=55, scale=15, size=1)[0], 10, 100))
                
                row['optimizer'] = 'Adam' if np.random.random() < 0.5 else 'SGD'
                
                # Learning Rate (Log-Normal distribution)
                row['learning_rate'] = round(np.random.lognormal(mean=-3.0, sigma=0.8, size=1)[0], 6)
                
                # --- Fixed Parameters ---
                row['store_weight'] = False
                row['status'] = '-'
                
                sweep_data.append(row)
            
    # 3. Finalization and Output
    df = pd.DataFrame(sweep_data)
    
    try:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully generated {len(df)} configurations and saved to '{output_csv_file}'.")
        
        # Display sample output to show permutation
        print("\nExample of how class permutations look in output:")
        print(df.head(NUM_PERMUTATIONS)[['combo_id', 'source_combo_id', 'class1', 'class2', 'class3']].to_markdown(index=False))

    except Exception as e:
        print(f"Error writing to file: {e}")

# --- Execution Block ---

if __name__ == "__main__":

    OUTPUT_FILE = os.path.join(os.getcwd(), 'data', 'new_sweep.csv')

    # Parameters for the generation
    # 60 total samples per combination / 6 class permutations = 10 unique HP sets per permutation
    NUM_HP_SAMPLES_PER_COMBO = 60 
    RANDOM_SEED = 42
    
    generate_hp_sweep_csv(
        combinations_data=CLASS_COMBINATIONS_DATA,
        output_csv_file=OUTPUT_FILE,
        num_hp_samples_per_combination=NUM_HP_SAMPLES_PER_COMBO,
        seed=RANDOM_SEED
    )
