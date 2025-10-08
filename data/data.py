import wandb
import pandas as pd
import os
from typing import Optional

def fetch_and_export_wandb_data(
    entity: str,
    project_name: str,
    download_dir: str = "ownloaded_artifacts",
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

if __name__ == "__main__":
    # Example usage with the original values
    entity = '25205269-stellenbosch-university'
    project_name = 'MSc_MLAI'
    
    # You can call the function with your specific project details
    df = fetch_and_export_wandb_data(entity, project_name)
    
    # You can now work with the DataFrame directly
    if not df.empty:
        print("\nHead of the generated DataFrame:")
        print(df.head())
