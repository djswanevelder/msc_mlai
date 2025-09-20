import wandb
import pandas as pd
import os

api = wandb.Api()

entity = '25205269-stellenbosch-university'
project_name = 'MSc_MLAI'

print(f"Fetching all runs from project: {project_name}...")
runs = api.runs(path=f"{entity}/{project_name}")

all_run_data = []

# Define the directory where artifacts will be saved
download_dir = "downloaded_artifacts"
os.makedirs(download_dir, exist_ok=True)
print(f"Artifacts will be downloaded to: {os.path.abspath(download_dir)}")

for i, run in enumerate(runs):
    print(f"Processing run {i+1}/{len(runs)}: {run.name}")

    # Extract basic run information
    run_info = {
        'run_id': run.id,
        'run_name': run.name,
        'creation_time': run.created_at,
        'state': run.state,
    }

    # Fetch and download artifacts
    artifacts = run.logged_artifacts()
    downloaded_artifacts = []
    
    if artifacts:
        print(f"Found {len(artifacts)} artifact(s) to download.")
        for artifact_version in artifacts:
            try:
                # The artifact.name is in the format "artifact_name:version"
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

    run_info['downloaded_artifacts'] = ", ".join(downloaded_artifacts)
    
    summary_data = run.summary._json_dict
    combined_data = {**run_info, **summary_data}
    all_run_data.append(combined_data)

print("\nData extraction complete. Creating DataFrame...")

df = pd.DataFrame(all_run_data)

output_file = "wandb_runs_data.csv"
df.to_csv(output_file, index=False)

print(f"Successfully exported run metadata to {output_file}")
