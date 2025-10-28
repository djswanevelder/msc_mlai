import wandb
import pandas as pd
import os
from typing import Optional
import shutil


def fetch_and_export_wandb_data(
    entity: str,
    project_name: str,
    download_dir: str = "downloaded_artifacts",
    output_csv_file: str = "wandb_runs_data.csv"
) -> pd.DataFrame:
    """
    Fetches run data and artifacts from a Weights & Biases project, downloads
    the artifacts, and exports the run metadata to a CSV file.

    Args:
        entity (str): The W&B entity name (e.g., username or team name).
        project_name (str): The W&B project name.
        download_dir (str, optional): The directory to download artifacts to.
            Defaults to "wandb_downloaded_artifacts".
        output_csv_file (str, optional): The name of the output CSV file.
            Defaults to "wandb_runs_data.csv".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the metadata for all runs.
    """
    # Initialize the W&B API
    api = wandb.Api()

    print(f"Fetching all runs from project: {project_name} under entity: {entity}...")
    try:
        runs = api.runs(path=f"{entity}/{project_name}")
    except Exception as e:
        print(f"Error fetching runs. Please check the entity and project name: {e}")
        return pd.DataFrame()

    all_run_data = []

    # Create the directory for artifacts if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    print(f"Artifacts will be downloaded to: {os.path.abspath(download_dir)}")

    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{len(runs)}: {run.name}")

        run_info = {
            'run_id': run.id,
            'run_name': run.name,
            'creation_time': run.created_at,
            'state': run.state,
        }

        # Fetch and download artifacts
        downloaded_artifacts = []
        try:
            artifacts = run.logged_artifacts()
            if artifacts:
                print(f"Found {len(artifacts)} artifact(s) to download.")
                for artifact_version in artifacts:
                    try:
                        # Construct the full artifact path and download
                        artifact_full_name = f"{entity}/{project_name}/{artifact_version.name}"
                        artifact_obj = api.artifact(artifact_full_name)

                        # Create a sub-directory for each artifact
                        artifact_download_path = os.path.join(download_dir, artifact_version.name.split(':')[0])
                        os.makedirs(artifact_download_path, exist_ok=True)

                        artifact_obj.download(root=artifact_download_path)
                        print(f"-> Successfully downloaded artifact: {artifact_version.name}")
                        downloaded_artifacts.append(artifact_version.name)
                    except Exception as e:
                        print(f"-> Failed to download artifact {artifact_version.name}: {e}")
        except Exception as e:
            print(f"Error retrieving artifacts for run {run.name}: {e}")
        
        run_info['downloaded_artifacts'] = ", ".join(downloaded_artifacts)
        
        # Combine run info with summary metrics
        summary_data = run.summary._json_dict
        combined_data = {**run_info, **summary_data}
        all_run_data.append(combined_data)

    print("\nData extraction complete. Creating DataFrame...")

    df = pd.DataFrame(all_run_data)

    # Export to CSV
    df.to_csv(output_csv_file, index=False)
    print(f"Successfully exported run metadata to {output_csv_file}")
    
    return df

def extract_checkpoint_weights(source_dir: Optional[str] = None):
    """
    Iterates through subfolders in the source_dir, looks for a .pth file 
    with the same name as the subfolder, and copies it to a central 
    'extracted_weights' directory.

    Args:
        source_dir (Optional[str]): The directory containing the model subfolders. 
                                    Defaults to the current working directory if None.
    """
    # Use the provided directory or default to the current working directory
    if source_dir is None:
        source_dir = os.getcwd()
    
    # Define the destination directory for the extracted .pth files.
    target_dir = os.path.join(os.path.dirname(source_dir), 'extracted_weights')

    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"--- Starting weight extraction ---")
    print(f"Source Directory: {source_dir}")
    print(f"Target Directory: {target_dir}")
    print("-" * 35)

    # Iterate over every item in the source directory
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        # 1. Check if the item is a directory (a model folder)
        # Also check that the directory is not the target directory we just created
        if os.path.isdir(item_path) and item_path != target_dir:
            # The expected .pth file name is the same as the folder name, plus the .pth extension
            pth_filename = f"{item}.pth"
            
            # Construct the full path to the expected checkpoint file inside the folder
            source_pth_path = os.path.join(item_path, pth_filename)
            
            # 2. Check if the .pth file actually exists
            if os.path.exists(source_pth_path):
                # Construct the path for the copied file in the target directory
                target_pth_path = os.path.join(target_dir, pth_filename)
                
                # 3. Copy the file to the central directory (shutil.copy2 preserves metadata)
                shutil.copy2(source_pth_path, target_pth_path)
                print(f"  [SUCCESS] Copied: {pth_filename}")
            else:
                # This handles folders that might not contain the expected file (e.g., run-*-history folders)
                print(f"  [SKIP] No .pth file found named '{pth_filename}' in '{item}'.")

    print("-" * 35)
    print(f"Extraction complete! All checkpoint files are now collected in the '{target_dir}' folder.")


if __name__ == "__main__":
    # # Example usage with the original values
    entity = '25205269-stellenbosch-university'
    project_name = 'MSc_MLAI'
    
    # You can call the function with your specific project details
    df = fetch_and_export_wandb_data(entity, project_name)
    
    # You can now work with the DataFrame directly
    if not df.empty:
        print("\nHead of the generated DataFrame:")
        print(df.head())

    extract_checkpoint_weights(source_dir='/home/dj/Desktop/msc_mlai/data/weights/downloaded_artifacts') 
    
    # Running without arguments will still default to the current directory:
    # extract_checkpoint_weights(source_dir='./weights/downloaded_artifacts')
