import wandb
import pandas as pd

# Initialize the Weights & Biases API
api = wandb.Api()

# Replace with your specific entity and project name
# Based on your example, these values are used directly.
entity = '25205269-stellenbosch-university'
project_name = 'MSc_MLAI'

# Get all runs from the specified project
print(f"Fetching all runs from project: {project_name}...")
runs = api.runs(path=f"{entity}/{project_name}")

# List to store data from each run
all_run_data = []

# Iterate through each run and extract relevant information
for i, run in enumerate(runs):
    print(f"Processing run {i+1}/{len(runs)}: {run.name}")

    # Extract basic run information
    run_info = {
        'run_id': run.id,
        'run_name': run.name,
        'creation_time': run.created_at,
        'state': run.state,
    }

    # Extract the names of any artifacts linked to the run
    # This returns a list of artifact objects
    artifacts = run.logged_artifacts()
    artifact_names = [artifact.name for artifact in artifacts]
    run_info['artifacts'] = ", ".join(artifact_names)

    # Extract the summary metrics and flatten the dictionary
    # The `_json_dict` method correctly converts the summary object to a dictionary.
    summary_data = run.summary._json_dict

    # Combine all data into a single dictionary
    combined_data = {**run_info, **summary_data}
    all_run_data.append(combined_data)

print("Data extraction complete. Creating DataFrame...")

# Create a pandas DataFrame from the collected data
df = pd.DataFrame(all_run_data)

# Export the DataFrame to a CSV file
output_file = "wandb_runs_data.csv"
df.to_csv(output_file, index=False)

print(f"Successfully exported data to {output_file}")
