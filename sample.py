import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sample_data_uniform(input_filepath: str, total_samples: int, output_filepath: str) -> None:
    """
    Samples data from a CSV file to achieve an as-close-as-possible uniform
    distribution over the 'average_similarity_score' column.

    The function first bins the data and then samples an equal number of
    records from each bin. This ensures that the final dataset has a
    relatively flat distribution of scores.

    Args:
        input_filepath (str): The path to the input CSV file.
        total_samples (int): The target number of samples to include in the output.
                             The actual number may be slightly less if some bins
                             do not contain enough data.
        output_filepath (str): The path to save the output CSV file.
    """
    # Try to read the CSV file. Handle the case where the file is not found.
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Check if the required column exists
    if 'average_similarity_score' not in df.columns:
        print("Error: The CSV file must contain a column named 'average_similarity_score'.")
        return

    # Find and print the min and max scores
    min_score = df['average_similarity_score'].min()
    max_score = df['average_similarity_score'].max()
    print(f"Min score in the dataset: {min_score}")
    print(f"Max score in the dataset: {max_score}")

    num_bins = 10
    samples_per_bin = total_samples // num_bins
    
    if samples_per_bin == 0:
        print(f"The total_samples value ({total_samples}) is too small for {num_bins} bins.")
        return

    df['bin'] = pd.cut(df['average_similarity_score'], bins=num_bins, labels=False)
    
    print("\nOriginal data distribution across bins:")
    print(df['bin'].value_counts().sort_index())

    sampled_bins = []
    
    for bin_label in df['bin'].unique():
        bin_data = df[df['bin'] == bin_label]
        
        num_to_sample = min(samples_per_bin, len(bin_data))
        
        if num_to_sample > 0:
            sampled_bins.append(bin_data.sample(n=num_to_sample, replace=False, random_state=42))

    if not sampled_bins:
        print("\nSampling failed. No data was found in the bins.")
        return
        
    sampled_df = pd.concat(sampled_bins).reset_index(drop=True)
    
    final_df = sampled_df.drop(columns=['bin']).sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(output_filepath, index=False)
    
    print(f"\nSampling complete. The new dataset with {len(final_df)} samples is saved to '{output_filepath}'.")
    
    final_df['bin'] = pd.cut(final_df['average_similarity_score'], bins=num_bins, labels=False)
    print("\nFinal sampled data distribution across bins:")
    print(final_df['bin'].value_counts().sort_index())

def plot_similarity_histogram(input_file,bins):
    """
    Plots a histogram of the specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
    """
    dataframe = pd.read_csv(input_file)
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['average_similarity_score'], bins=bins, edgecolor='black')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('similarity_score_histogram.png')

if __name__ == "__main__":
    sample_data_uniform('permutations.csv', 2000, 'sweep.csv')
    np.random.seed(42)
    plot_similarity_histogram('sweep.csv', bins = 100)
    # plot_similarity_histogram('permutations.csv', bins = 100)

    df = pd.read_csv('sweep.csv')
    df['score'] = df['average_similarity_score'].round(4)
    df = df.drop('average_similarity_score',axis=1)

    # Add a seed to each line
    df['seed'] = np.random.randint(0, 10000, size=len(df))
    df['seed'] = df['seed'].apply(lambda x: '%04d' % x)


    df['early_epoch'] = np.random.randint(1, 5, size=len(df))
    # df['max_epoch'] = np.random.randint(100, 350, size=len(df)) # maybe make a guassian?
    df['max_epoch'] =  np.random.normal(loc=150, scale=15, size=len(df)).astype(int)
    df['store_weight'] = np.random.random(len(df)) < 0.3
    df['optimizer'] = np.where(np.random.random(len(df)) < 0.5, 'Adam', 'SGD')

    # df_adam = df.copy()
    # df_sgd = df.copy()

    # df_adam['optimizer'] = 'Adam'
    # df_sgd['optimizer'] = 'SGD'
    # new_df = pd.concat([df_adam, df_sgd], ignore_index=True)

    df.to_csv('sweep.csv', index=False)
