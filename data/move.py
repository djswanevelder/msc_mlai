import pandas as pd
import os
import shutil
from tqdm import tqdm
import ast # Necessary to safely convert the string representation of a list

def move_selected_weights(csv_path: str, source_dir: str, target_dir: str):
    """
    Reads a CSV, extracts weight filenames from the 'artifacts' column (which 
    contains a string representation of a list), and moves those files 
    from a source directory to a target directory.
    """
    
    # The column containing the list of weight filenames
    ARTIFACTS_COLUMN = 'artifacts' 

    try:
        # 1. Load the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if ARTIFACTS_COLUMN not in df.columns:
        print(f"❌ Error: The required column '{ARTIFACTS_COLUMN}' was not found in the CSV.")
        return

    # 2. Prepare the directories
    if not os.path.isdir(source_dir):
        print(f"❌ Error: Source directory not found at {source_dir}")
        return
        
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory ensured: {target_dir}")

    moved_count = 0
    print("\nStarting file movement...")
    
    # 3. Iterate through runs and move files
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Moving Weight Files"):
        
        artifacts_str = row[ARTIFACTS_COLUMN]
        
        if pd.isna(artifacts_str):
            continue
            
        try:
            # Safely convert the string "['file1.pth', 'file2.pth']" into a list
            weight_filenames = ast.literal_eval(artifacts_str)
            
            # Ensure we're iterating over an iterable (like a list)
            if not isinstance(weight_filenames, list):
                print(f"\n⚠️ Warning: Artifacts entry for run {row.get('Name', index)} is not a list and was skipped.")
                continue

            # Iterate over each filename in the extracted list
            for weight_filename in weight_filenames:
                if pd.notna(weight_filename) and isinstance(weight_filename, str):
                    source_file = os.path.join(source_dir, weight_filename)
                    target_file = os.path.join(target_dir, weight_filename)
                    
                    if os.path.exists(source_file):
                        shutil.move(source_file, target_file)
                        moved_count += 1
                    else:
                        print(f"\n⚠️ Warning: File not found in source for run {row.get('Name', index)}: {weight_filename}")
                        
        except ValueError as e:
            print(f"\n❌ Error converting artifact string for run {row.get('Name', index)}: {artifacts_str}. Error: {e}")
        except Exception as e:
            print(f"\n❌ An error occurred during move for run {row.get('Name', index)}: {e}")

    print("\n--- Process Complete ---")
    print(f"Total files moved: {moved_count}")

# --- Example Execution ---
if __name__ == '__main__':
    
    # 🚨 CONFIGURE THESE PATHS AND FILENAMES 🚨
    # Assuming 'data/meta_data.csv' is the output from the previous script
    CSV_FILE = 'data/wandb_w_artifacts.csv' 
    SOURCE_DIRECTORY = 'data/weights/' 
    TARGET_DIRECTORY = 'data/weights/selected/' 
    
    move_selected_weights(CSV_FILE, SOURCE_DIRECTORY, TARGET_DIRECTORY)