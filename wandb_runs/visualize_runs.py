import pandas as pd
import matplotlib.pyplot as plt

# Define the name of the input CSV file
input_file = "runs.csv"
bins = 30
# Use a try-except block to handle file not found errors
try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    print(f"Successfully loaded data from {input_file}")

    # --- Data Visualization ---

    # Create a figure and axes for the plot
    plt.figure(figsize=(10, 6))

    # Plot a histogram of the 'val_loss' column
    # The `alpha` parameter makes the histograms semi-transparent so you can see both
    plt.hist(df['val_loss'], bins=bins, alpha=0.7, label='Validation Loss')

    # Plot a histogram of the 'train_loss' column
    plt.hist(df['val_loss_at_early_epoch'], bins=bins, alpha=0.7, label='Early Epoch Validation Loss')

    # Add a horizontal line at 1.0986 for the three-class random guess baseline
    plt.axvline(1.0986, color='red', linestyle='--', linewidth=2, label='Random Guess Loss (3 classes)')

    # Add a title and labels to the plot for clarity
    plt.title('Distribution of Validation and Training Loss')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency of Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure you have run the 'export_runs_to_csv.py' script first.")
except KeyError as e:
    print(f"Error: A required column was not found in the CSV file. {e}")
    print("Please check the column names in your CSV file.")
