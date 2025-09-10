import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_score_loss_scatter(file_path):
    """
    Reads a CSV file and creates a scatter plot of score vs. validation loss and
    score vs. early epoch validation loss, with a line of best fit for each.

    Args:
        file_path (str): The path to the input CSV file.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")

    # Check for required columns
    required_columns = ['distance_score', 'val_loss', 'val_loss_at_early_epoch']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: The following columns are required for plotting but were not found: {missing_cols}")
        return

    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))

    # --- Plot the Scatter Points ---
    # Scatter plot for the final val_loss (in red)
    plt.scatter(
        df['distance_score'], 
        df['val_loss'], 
        color='red', 
        label='Final Validation Loss',
        alpha=0.7
    )

    # Scatter plot for the early val_loss (in blue)
    plt.scatter(
        df['distance_score'], 
        df['val_loss_at_early_epoch'], 
        color='blue', 
        label='Early Validation Loss',
        alpha=0.7
    )

    # --- Calculate and Plot Lines of Best Fit ---
    # Calculate the line of best fit for final validation loss
    m_final, c_final = np.polyfit(df['score'], df['val_loss'], 1)
    plt.plot(
        df['score'], 
        m_final * df['score'] + c_final, 
        color='darkred', 
        linestyle='--', 
        label='Final Loss Trend Line'
    )

    # Calculate the line of best fit for early validation loss
    m_early, c_early = np.polyfit(df['score'], df['val_loss_at_early_epoch'], 1)
    plt.plot(
        df['score'], 
        m_early * df['score'] + c_early, 
        color='darkblue', 
        linestyle='--', 
        label='Early Loss Trend Line'
    )

    # Add titles and labels
    plt.title('Score vs. Validation Loss Scatter Plot')
    plt.xlabel('Score')
    plt.ylabel('Loss')
    
    # Add a legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()

def plot_loss_histograms(file_path, num_bins):
    """
    Reads a CSV file and plots histograms of validation loss and early epoch validation loss.

    Args:
        file_path (str): The path to the input CSV file.
        num_bins (int): The number of bins for the histogram.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")

    # --- Data Visualization ---

    # Create a figure and axes for the plot
    plt.figure(figsize=(10, 6))

    # Plot a histogram of the 'val_loss' column
    # The `alpha` parameter makes the histograms semi-transparent so you can see both
    plt.hist(df['val_loss'], bins=num_bins, alpha=0.7, label='Validation Loss')

    # Plot a histogram of the 'val_loss_at_early_epoch' column
    plt.hist(df['val_loss_at_early_epoch'], bins=num_bins, alpha=0.7, label='Early Epoch Validation Loss')

    # Add a horizontal line at 1.0986 for the three-class random guess baseline
    plt.axvline(1.0986, color='red', linestyle='--', linewidth=2, label='Random Guess Loss (3 classes)')

    # Add a title and labels to the plot for clarity
    plt.title('Distribution of Validation and Early Epoch Validation Loss')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency of Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()

# Define the name of the input CSV file and number of bins
input_file = "runs.csv"
bins = 30
plot_loss_histograms(input_file,bins)
# Use a try-except block to handle potential errors
try:
    plot_score_loss_scatter(input_file)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure you have the 'runs.csv' file in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
