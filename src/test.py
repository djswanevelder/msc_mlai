import pandas as pd
import os

# Define the filename and the columns required for calculation
CSV_FILE = 'data/encoding_comparison_stats.csv'
COLUMNS_TO_ANALYZE = ['agreement', 'cos_sim', 'avg_correlation']

def generate_mock_data(filename):
    """
    Generates a sample CSV file if it doesn't exist, using the required column headings, 
    for immediate testing purposes.
    """
    data = {
        'rec_mse': [0.01, 0.02, 0.015, 0.03, 0.005],
        'agreement': [0.85, 0.78, 0.91, 0.65, 0.95],
        'original_accuracy': [0.92, 0.91, 0.93, 0.88, 0.94],
        'reconstructed_accuracy': [0.91, 0.89, 0.92, 0.85, 0.95],
        'cos_sim': [0.98, 0.89, 0.95, 0.82, 0.99],
        'avg_correlation': [0.75, 0.62, 0.88, 0.55, 0.91],
        'best_agreement': [0.86, 0.79, 0.93, 0.68, 0.96],
        'best_rec_accuracy': [0.92, 0.90, 0.94, 0.87, 0.95]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"INFO: Generated mock data file '{filename}' because it was not found.")


def analyze_csv_data(filename, columns):
    """
    Reads a CSV file, calculates the mean, standard deviation, min, and max for specified columns,
    and prints the results in a formatted table.
    """
    if not os.path.exists(filename):
        print(f"WARNING: File '{filename}' not found. Generating mock data to run the script.")
        generate_mock_data(filename)

    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(filename)
        print(f"Successfully loaded data from '{filename}'.")

        # Check if all required columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns in the CSV: {', '.join(missing_cols)}. Calculations will be skipped for these columns.")
            # Proceed only with the columns that are present
            columns = [col for col in columns if col not in missing_cols]
            if not columns:
                print("No valid columns left to analyze. Exiting.")
                return

        # Calculate statistics (mean, standard deviation, min, max) for the specified columns
        # We include 'std' (Standard Deviation) as the measure of distribution/spread.
        results = df[columns].agg(['mean', 'std', 'min', 'max'])

        print("\n--- Statistical Analysis Results ---")
        # Print the results in a nicely formatted Markdown table
        print(results.to_markdown(numalign="left", stralign="left"))
        print("-" * 40)

    except Exception as e:
        print(f"An error occurred during file reading or calculation. Please ensure pandas is installed (pip install pandas). Error: {e}")

if __name__ == "__main__":
    analyze_csv_data(CSV_FILE, COLUMNS_TO_ANALYZE)
