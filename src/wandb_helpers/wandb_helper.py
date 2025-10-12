import wandb
import pandas as pd
import os
from typing import List, Dict, Any, Tuple, Optional

API_CLIENT: Optional[wandb.Api] = None

def initialize_and_setup(entity: str, project_name: str, download_dir: str) -> Tuple[wandb.Api, wandb.apis.public.Runs]:
    """
    Initializes the W&B API and sets up the local artifact download directory.

    Args:
        entity (str): The W&B username or team name.
        project_name (str): The name of the W&B project.
        download_dir (str): The local directory path for artifacts.

    Returns:
        tuple: A tuple containing:
               - api (wandb.Api): The initialized W&B API client.
               - runs (wandb.apis.public.Runs): Collection of all runs for the project.
    """
    global API_CLIENT
    api = wandb.Api()
    API_CLIENT = api
    print(f"Fetching all runs from project: {project_name}...")
    runs = api.runs(path=f"{entity}/{project_name}")

    os.makedirs(download_dir, exist_ok=True)
    print(f"Artifacts will be downloaded to: {os.path.abspath(download_dir)}")

    return api, runs

def extract_config_info(csv_filepath: str) -> pd.DataFrame:
    """
    Reads an external CSV file containing run-specific configuration or metadata.

    Args:
        csv_filepath (str): The path to the external CSV file (e.g., 'sweep.csv').

    Returns:
        pd.DataFrame: A DataFrame containing the external run metadata.
    """
    try:
        external_df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded external config data from {csv_filepath} with {len(external_df)} rows.")
        return external_df
    except FileNotFoundError:
        print(f"Error: External config file not found at {csv_filepath}. Returning empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading external config file {csv_filepath}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

def get_specific_epoch_metric(run: wandb.apis.public.Run, metric_key: str, target_epoch: int) -> Optional[float]:
    """
    Retrieves the value of a specific metric (e.g., 'val_loss') at a given epoch.

    Args:
        run (wandb.apis.public.Run): The W&B run object.
        metric_key (str): The name of the metric to retrieve (e.g., 'val_loss').
        target_epoch (int): The epoch number at which to retrieve the metric.

    Returns:
        Optional[float]: The metric value at the target epoch, or None if not found.
    """
    try:
        # Fetch history data, specifically 'epoch' and the required metric
        history = run.history(keys=['epoch', metric_key])

        # Filter the history to find the row corresponding to the target epoch
        result = history[history['epoch'] == target_epoch]

        if not result.empty:
            # Return the metric value from the first matching row
            return result[metric_key].iloc[0]
    except Exception as e:
        # Suppress detailed history fetch warnings here to keep the post-processing clean
        return None
    
    return None

def process_runs_and_export(api: wandb.Api, runs: wandb.apis.public.Runs, external_data_df: pd.DataFrame, entity: str, project_name: str, download_dir: str, output_file: str, columns_to_extract: List[str], download_artifacts: bool):
    """
    Iterates through W&B runs, extracts metadata, merges it with external data,
    (optionally) downloads artifacts, and exports the filtered data to a CSV file.
    """
    all_run_data = []
    total_runs = len(runs)
    external_data_available = not external_data_df.empty

    if external_data_available and total_runs != len(external_data_df):
        print(f"Warning: W&B run count ({total_runs}) does not match external data count ({len(external_data_df)}). Data alignment may be incorrect.")

    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{total_runs}: {run.name}")

        # 1. Fetch and process artifacts
        artifacts = run.logged_artifacts()
        
        processed_artifact_names = []
        for artifact in artifacts:
            name = artifact.name
            if "history" in name.lower():
                continue
            if ":" in name:
                name = name.split(':')[0]
            if not name.endswith('.pth'):
                 name += '.pth'
            processed_artifact_names.append(name)
            
        logged_artifact_names = ", ".join(processed_artifact_names)
        downloaded_artifact_names = []
        
        # Merge W&B data
        summary_data = run.summary._json_dict
        config_data = run.config

        # 2. Start combined data dictionary with W&B info and data
        combined_data: Dict[str, Any] = {
            'run_id': run.id,
            'run_name': run.name,
            'creation_time': run.created_at,
            'state': run.state,
            'logged_artifacts_raw': logged_artifact_names, # Maps to 'weights' column
            **summary_data,
            **config_data
        }
        
        # 3. Merge External Data (Order-based integration)
        if external_data_available and i < len(external_data_df):
            external_row = external_data_df.iloc[i].to_dict()
            
            for key in ['class1', 'class2', 'class3']:
                if key in external_row and isinstance(external_row[key], str):
                    external_row[key] = external_row[key].lower()
            
            combined_data.update(external_row)

        # 4. Artifact Downloading (Optional)
        if download_artifacts:
            print(f"Artifact downloading is ENABLED.")
            if artifacts:
                downloadable_artifacts = [a for a in artifacts if "history" not in a.name.lower()]
                
                print(f"Found {len(downloadable_artifacts)} artifact(s) to download after filtering.")
                for artifact_version in downloadable_artifacts:
                    try:
                        artifact_full_name = f"{entity}/{project_name}/{artifact_version.name}"
                        artifact_obj = api.artifact(artifact_full_name)
                        base_artifact_name = artifact_version.name.split(':')[0]
                        artifact_download_path = os.path.join(download_dir, base_artifact_name)
                        os.makedirs(artifact_download_path, exist_ok=True)
                        artifact_obj.download(root=artifact_download_path)
                        downloaded_artifact_names.append(base_artifact_name + '.pth')
                    except Exception as e:
                        print(f"-> Failed to download artifact {artifact_version.name}: {e}")
        else:
             print(f"Artifact downloading is DISABLED.")


        # 5. Finalize Data
        combined_data['downloaded_artifacts'] = ", ".join(downloaded_artifact_names)
        all_run_data.append(combined_data)

    print("\nData extraction complete. Creating DataFrame...")

    # Export results
    df = pd.DataFrame(all_run_data)

    column_mapping = {'weights': 'logged_artifacts_raw'}
    mapped_cols = [column_mapping.get(col, col) for col in columns_to_extract]

    valid_cols = [col for col in mapped_cols if col in df.columns]

    if valid_cols:
        df = df[valid_cols]
        if 'logged_artifacts_raw' in valid_cols and 'weights' in columns_to_extract:
            df.rename(columns={'logged_artifacts_raw': 'weights'}, inplace=True)
    else:
        print("Warning: None of the specified columns were found in the run data.")

    df.to_csv(output_file, index=False)
    print(f"Successfully exported run metadata to {output_file}.")


def post_process_early_epoch_loss(output_file: str, entity: str, project_name: str, target_metric: str = 'val_loss', early_epoch_col: str = 'early_epoch', new_metric_col: str = 'early_epoch_val_loss'):
    """
    Loads the saved CSV, retrieves the specific metric (val_loss) at the early_epoch + 1
    from W&B history for each run, and updates the CSV with the new column.
    """
    global API_CLIENT
    if API_CLIENT is None:
        print("Error: W&B API not initialized. Cannot fetch history data.")
        return

    try:
        df = pd.read_csv(output_file)
        print(f"\nStarting post-processing for {len(df)} runs to find {new_metric_col}...")
    except FileNotFoundError:
        print(f"Error: Output CSV file not found at {output_file}.")
        return

    # Add the new column, initializing to None
    df[new_metric_col] = None

    for index, row in df.iterrows():
        run_id = row['run_id']
        early_epoch_val = row.get(early_epoch_col)

        if early_epoch_val is None or not isinstance(early_epoch_val, (int, float)):
            df.loc[index, new_metric_col] = None
            continue

        try:
            run = API_CLIENT.run(f"{entity}/{project_name}/{run_id}")
            target_epoch = int(early_epoch_val) + 1
            # print(f"-> Fetching {target_metric} at epoch {target_epoch} for run {run.name} ({index + 1}/{len(df)})...")
            
            metric_value = get_specific_epoch_metric(run, target_metric, target_epoch)
            df.loc[index, new_metric_col] = metric_value

        except Exception as e:
            df.loc[index, new_metric_col] = None
            print(f"Error processing run {run_id} for history: {e}")

    df.to_csv(output_file, index=False)
    print(f"Post-processing complete. Updated data exported to {output_file}.")


def normalize_artifacts_to_rows(input_file: str, output_file: str):
    """
    Transforms the run-centric CSV into an artifact-centric CSV.

    Each row in the input (run) becomes two rows in the output (two artifacts), 
    assigning the early stop loss to the first artifact and the final loss to the second.

    Args:
        input_file (str): Path to the CSV file created by process_runs_and_export.
        output_file (str): Path to the new CSV file to save the normalized data.
    """
    print(f"\nStarting normalization: transforming runs into individual artifact rows...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_file}. Aborting normalization.")
        return

    normalized_rows = []
    
    # Required columns from the input CSV
    required_cols = ['run_id', 'run_name', 'class1', 'class2', 'class3', 'weights', 'val_loss', 'early_epoch_val_loss']

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in input CSV for normalization: {missing_cols}. Aborting.")
        return

    for _, row in df.iterrows():
        base_data = {
            'run_id': row['run_id'],
            'run_name': row['run_name'],
            'class1': row['class1'],
            'class2': row['class2'],
            'class3': row['class3'],
        }

        # Safely split the 'weights' string
        weights_list = [w.strip() for w in str(row['weights']).split(',') if w.strip()]
        
        if len(weights_list) >= 2:
            # 1. Early Stop Artifact (Artifact 1)
            normalized_rows.append({
                **base_data,
                'artifact_name': weights_list[0],
                'val_loss': row['early_epoch_val_loss']
            })

            # 2. Final Artifact (Artifact 2)
            normalized_rows.append({
                **base_data,
                'artifact_name': weights_list[1],
                'val_loss': row['val_loss']
            })
        elif len(weights_list) == 1:
            print(f"Warning: Only one artifact found for run {row['run_name']}. Using final loss.")
            normalized_rows.append({
                **base_data,
                'artifact_name': weights_list[0],
                'val_loss': row['val_loss']
            })
        else:
            print(f"Warning: No valid artifacts found for run {row['run_name']}. Skipping.")
    
    if normalized_rows:
        normalized_df = pd.DataFrame(normalized_rows)
        # Ensure the final columns are in the requested order
        final_cols = ['run_id', 'run_name', 'class1', 'class2', 'class3', 'artifact_name', 'val_loss']
        normalized_df = normalized_df[final_cols]
        
        normalized_df.to_csv(output_file, index=False)
        print(f"Normalization complete. Exported {len(normalized_df)} artifact rows to {output_file}.")
    else:
        print("No rows generated during normalization.")



if __name__ =='__main__':

    ENTITY = '25205269-stellenbosch-university'
    PROJECT_NAME = 'MSc_MLAI'
    DOWNLOAD_DIR = "downloaded_artifacts"
    OUTPUT_FILE = "wandb_runs_data.csv"
    NORMALIZED_OUTPUT_FILE = "meta-dataset_info.csv"
    SWEEP_CSV_FILE = "data/sweep.csv"

    COLUMNS_TO_EXTRACT = [
        'run_id', 
        'run_name',
        'class1', 
        'class2', 
        'class3', 
        'early_epoch', 
        'max_epoch',
        'val_loss',
        'weights',
        'downloaded_artifacts',
    ]
    DOWNLOAD_ARTIFACTS = False 

    # api_client, all_runs = initialize_and_setup(ENTITY, PROJECT_NAME, DOWNLOAD_DIR)
    # external_df = extract_config_info(SWEEP_CSV_FILE)
    # process_runs_and_export(api_client, all_runs, external_df, ENTITY, PROJECT_NAME, DOWNLOAD_DIR, OUTPUT_FILE, COLUMNS_TO_EXTRACT, DOWNLOAD_ARTIFACTS)
    # post_process_early_epoch_loss(OUTPUT_FILE, ENTITY, PROJECT_NAME)

    normalize_artifacts_to_rows(OUTPUT_FILE, NORMALIZED_OUTPUT_FILE)