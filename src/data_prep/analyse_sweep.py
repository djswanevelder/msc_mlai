import pandas as pd
import os
from collections import Counter

def analyze_sweep_permutations(input_csv_file: str) -> None:
    """
    Analyzes the sweep.csv file to count permutations (class sets) 
    that have been repeated and which individual classes appear most frequently.

    Args:
        input_csv_file: Path to the generated sweep CSV file (e.g., 'data/sweep.csv').
    """
    print(f"--- Starting Analysis of: {input_csv_file} ---")
    
    if not os.path.exists(input_csv_file):
        print(f"Error: The input sweep file '{input_csv_file}' was not found.")
        print("Please ensure the generation script has been run first.")
        return

    try:
        df = pd.read_csv(input_csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    total_permutations = len(df)
    
    # 1. Identify and count repeated permutations (set of classes)
    
    # Create a new column with the sorted tuple of the three classes.
    # Sorting ensures that ('dog', 'cat', 'bird') is treated the same as 
    # ('cat', 'dog', 'bird') because they are the same set of classes for training.
    df['class_set'] = df[['class1', 'class2', 'class3']].apply(
        lambda x: tuple(sorted(x.astype(str).tolist())), axis=1
    )

    # Count the frequency of each unique class set
    set_counts = df['class_set'].value_counts()
    
    # Identify repeated sets (counts > 1)
    repeated_sets = set_counts[set_counts > 1]
    
    # Calculate the total number of duplicate rows
    # This is the sum of (count - 1) for every repeated set
    total_repetitions = (repeated_sets - 1).sum()
    
    # --- Repetition Analysis Output ---
    
    print("\n## 📊 Permutation Repetition Analysis 📊")
    print(f"Total configurations generated: {total_permutations}")
    print(f"Total unique class sets found: {len(set_counts)}")
    print(f"Number of sets repeated: {len(repeated_sets)}")
    print(f"Total number of repeated rows (redundancy): {total_repetitions}")

    if not repeated_sets.empty:
        print("\n### Top 5 Most Repeated Class Sets:")
        for (class_tuple, count) in repeated_sets.head(5).items():
            classes = ', '.join(class_tuple)
            print(f"- Set: ({classes}) was repeated {count} times.")
    else:
        print("\nNo repeated class sets were found in the sweep file.")

    # 2. Individual Class Frequency Analysis
    
    # Stack the three class columns into a single Series for easy counting
    all_classes = pd.concat([df['class1'], df['class2'], df['class3']])
    class_frequencies = all_classes.value_counts()
    
    # --- Individual Class Output ---
    
    print("\n## 🔍 Individual Class Frequency Analysis 🔍")
    print(f"Unique individual classes used: {len(class_frequencies)}")
    
    print("\n### Top 10 Most Frequently Used Classes:")
    for (class_name, count) in class_frequencies.head(10).items():
        print(f"- Class: '{class_name}' appeared {count} times.")
        
    print("\n### 10 Least Frequently Used Classes:")
    for (class_name, count) in class_frequencies.tail(10).items():
        print(f"- Class: '{class_name}' appeared {count} times.")
        
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    # Assuming 'sweep.csv' is saved in the 'data' folder next to this script's directory structure
    OUTPUT_FILE = os.path.join(os.getcwd(), 'data', 'sweep.csv')
    
    analyze_sweep_permutations(OUTPUT_FILE)