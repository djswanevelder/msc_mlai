import pandas as pd

# Define the names of the input and output CSV files
# IMPORTANT: Update these filenames to match your files
first_file = "wandb_runs_data.csv"
second_file = "../sweep.csv"
output_file = "runs.csv"

# Use a try-except block to handle file not found errors
try:
    # Read the two CSV files into pandas DataFrames
    df_first = pd.read_csv(first_file)
    df_second = pd.read_csv(second_file)

    print(f"Successfully loaded data from '{first_file}' and '{second_file}'.")

    # Verify that the number of rows matches before joining
    if len(df_first) != len(df_second):
        print(f"Warning: The number of rows in '{first_file}' ({len(df_first)}) "
              f"does not match the number of rows in '{second_file}' ({len(df_second)}).")
        print("The data may not align correctly.")

    # Join the two DataFrames line by line. This assumes they are in the same order.
    # The 'axis=1' argument performs a column-wise concatenation.
    df_merged = pd.concat([df_first, df_second], axis=1)

    # Export the merged DataFrame to a new CSV file
    df_merged.to_csv(output_file, index=False)
    print(f"Successfully merged and exported data to '{output_file}'.")

except FileNotFoundError as e:
    print(f"Error: The file '{e.filename}' was not found.")
    print("Please make sure both CSV files are in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
