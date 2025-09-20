import pandas as pd
import wandb
import numpy as np

# Initialize the Weights & Biases API
api = wandb.Api()

# Define the name of the input CSV file
input_file = "runs.csv"

# Use a try-except block to handle potential errors
try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    print(f"Successfully loaded data from '{input_file}'.")

    # Check if the required columns exist in the DataFrame
    required_columns = ['run_id', 'early_epoch']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The following columns are required but not found: "
              f"{[col for col in required_columns if col not in df.columns]}")
    else:
        # Add a new column to the DataFrame to store the fetched validation loss
        df['val_loss_at_early_epoch'] = np.nan

        # Loop through each row of the DataFrame
        for index, row in df.iterrows():
            run_id = row['run_id']
            run_name = row['run_name']
            early_epoch = row['early_epoch']
            
            # Fetch the run from the WandB API
            try:
                run = api.run(f"25205269-stellenbosch-university/MSc_MLAI/{run_id}")
                
                # Get the history of the run as a pandas DataFrame
                history_df = pd.DataFrame(run.history())
                
                # Find the val_loss at the specified early_epoch
                if 'epoch' in history_df.columns and 'val_loss' in history_df.columns:
                    val_loss_at_epoch = history_df[history_df['epoch'] == early_epoch]['val_loss']

                    if not val_loss_at_epoch.empty:
                        # Store the value in the new DataFrame column
                        df.loc[index, 'val_loss_at_early_epoch'] = val_loss_at_epoch.dropna().iloc[0]
                        print(f"Fetched val_loss for run {run_name}: {val_loss_at_epoch.dropna().iloc[0]:.4f}")

                    else:
                        print(f"Data for early_epoch {early_epoch} not found for run {run_name}. Setting to NaN.")

                else:
                    print(f"'epoch' or 'val_loss' not found in history for run {run_name}. Setting to NaN.")

            except Exception as e:
                print(f"Could not fetch data for run {run_name}. Error: {e}")
            
        # Save the updated DataFrame back to the CSV file
        df.to_csv(input_file, index=False)
        print(f"\nSuccessfully updated '{input_file}' with a new column.")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
