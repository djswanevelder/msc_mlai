import pandas as pd
import ast
import re
import os
from tqdm import tqdm
import wandb
from typing import Tuple, Any

# --- Configuration for WandB API ---
ENTITY = "25205269-stellenbosch-university" 
PROJECT_NAME = "MSc_MLAI"
WANDB_PROJECT = f"{ENTITY}/{PROJECT_NAME}"
# -----------------------------------

# Initialize tqdm for Pandas apply operations to show progress
tqdm.pandas(desc="Fetching Epoch Metrics from WandB")


def fetch_epoch_metrics_or_preserve_original(row: pd.Series, api: wandb.Api, wandb_project: str) -> Tuple[Any, Any]:
    """
    Fetches val_accuracy and val_loss from WandB history for a specific artifact_epoch.
    If the artifact_epoch is different from the run's final logged epoch, 
    it returns the original values from the CSV.
    """
    run_id = row['ID']
    artifact_epoch = row['artifact_epoch']
    
    # Preserve original CSV values by default
    original_val_acc = row['original_val_accuracy']
    original_val_loss = row['original_val_loss']
    
    # Check if the epoch is valid (not null)
    if pd.isna(artifact_epoch) or not isinstance(artifact_epoch, int):
        return original_val_acc, original_val_loss
    
    try:
        run = api.run(f"{wandb_project}/{run_id}")
        
        # 1. Check if the artifact_epoch matches the run's final epoch (which is usually the value logged in the summary/CSV)
        # Note: We assume the run's final logged epoch is stored in the 'epoch' column of the input CSV.
        final_epoch = row['epoch'] 
        
        if artifact_epoch == final_epoch:
            # If the artifact epoch matches the final run epoch, keep the original values (which are correct)
            return original_val_acc, original_val_loss
            
        # 2. If the epochs differ, we need to look up the specific history
        history = run.scan_history(keys=["val_accuracy", "val_loss", "epoch"])
        
        target_metrics = next(
            (
                h for h in history 
                if h.get('epoch') == artifact_epoch
            ), 
            None
        )
        
        if target_metrics:
            # Return the specific metrics found in the history for this artifact_epoch
            return target_metrics.get('val_accuracy'), target_metrics.get('val_loss')
        else:
            # If the specific epoch wasn't found in history, return NA/original (safest to return original)
            return original_val_acc, original_val_loss
            
    except Exception:
        # On any API error, return the original metrics to preserve data
        return original_val_acc, original_val_loss


def transform_runs_to_concise_artifacts(csv_path: str, output_path: str):
    """
    Reads run data, explodes by artifact, and conditionally fetches/preserves
    metrics, then saves the final cleaned CSV.
    """
    
    api = wandb.Api()

    # 1. READ AND EXPLODE THE INPUT CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found at {csv_path}")
        return
    
    if 'artifacts' not in df.columns:
        print("❌ Error: 'artifacts' column not found.")
        return

    # Store original metrics before explosion for preservation
    df['original_val_accuracy'] = df['val_accuracy']
    df['original_val_loss'] = df['val_loss']

    # Convert artifact string to list and explode
    df['artifacts'] = df['artifacts'].apply(ast.literal_eval)
    df = df.explode('artifacts', ignore_index=True)
    df = df.rename(columns={'artifacts': 'Artifact_Filename'})

    # 2. CLEAN AND PREPARE COLUMNS
    
    df = df.drop(columns=['End Time'], errors='ignore')
    
    def extract_classes(class_str):
        if pd.isna(class_str):
            return [None, None, None]
        classes = re.sub(r'[\[\]]', '', str(class_str)).split(',')
        return (classes + [None] * 3)[:3]

    class_data = df['data.classes'].apply(extract_classes).to_list()
    df[['class1', 'class2', 'class3']] = pd.DataFrame(class_data, index=df.index)
    
    # Extract artifact epoch number
    epoch_regex = r'_(\d+)\.pth$'
    df['artifact_epoch'] = df['Artifact_Filename'].str.extract(epoch_regex, expand=False).astype('Int64')
    
    # 3. CONDITIONAL METRIC FETCH/PRESERVATION VIA WANDB API
    print("Fetching specific epoch metrics or preserving original values...")
    
    # The result of this apply operation conditionally overwrites/preserves val_accuracy/val_loss
    df[['val_accuracy', 'val_loss']] = df.progress_apply(
        lambda row: fetch_epoch_metrics_or_preserve_original(row, api, WANDB_PROJECT), 
        axis=1, 
        result_type='expand'
    )
    
    # 4. SELECT AND RENAME FINAL COLUMNS (Dropping temporary columns)
    final_df = df[[
        'ID',
        'Name',
        'class1',
        'class2',
        'class3',
        'Artifact_Filename',
        'artifact_epoch', 
        'val_loss',
        'val_accuracy'
    ]].copy()

    final_df.columns = [
        'run_id', 'run_name', 'class1', 'class2', 'class3', 'artifact_name', 'artifact_epoch', 'val_loss', 'val_accuracy'
    ]
    
    # 5. SAVE THE RESULT TO CSV WITH HEADERS
    final_df.to_csv(output_path, index=False, header=True)
    
    print(f"\n✅ Data processed and saved to CSV with headers at: {output_path}")

# --- Execution ---
if __name__ == '__main__':
    CSV_INPUT_PATH = 'data/wandb_w_artifacts.csv' 
    CSV_OUTPUT_PATH = 'data/meta_data.csv' 
    
    transform_runs_to_concise_artifacts(CSV_INPUT_PATH, CSV_OUTPUT_PATH)