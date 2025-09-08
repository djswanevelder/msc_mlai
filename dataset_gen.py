from omegaconf import OmegaConf
from download import download
from itertools import combinations
import os, csv, random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
# nltk.download('wordnet')
from word2vec import measure_similarity, load_model

def calculate_path_similarity(class1_name: str, class2_name: str):
    """
    Args:
        class1_name: The name of the first class (e.g., 'dog').
        class2_name: The name of the second class (e.g., 'house').
        
    Returns:
        The path similarity score, a float between 0 and 1. 
        Returns None if a synset cannot be found for either class.
    """
    try:
        synset1 = wn.synsets(class1_name, pos=wn.NOUN)[0]
        synset2 = wn.synsets(class2_name, pos=wn.NOUN)[0]

        return synset1.path_similarity(synset2)

    except IndexError:
        print(f"Could not find a WordNet synset for one of the classes. Please check the spelling or specify a more detailed term.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def calculate_average_similarity_of_three_classes(class_list: list):
    """
    Calculates the average path similarity for a set of three classes.
    It computes the similarity for all three pairs and then averages them.

    Args:
        class_list: A list containing exactly three class names (strings).
        
    Returns:
        The average similarity score, or None if any synset is not found.
    """
    if len(class_list) != 3:
        print("Error: This function requires exactly three classes in the list.")
        return None

    class1, class2, class3 = class_list[0], class_list[1], class_list[2]

    sim_ab = calculate_path_similarity(class1, class2)
    sim_ac = calculate_path_similarity(class1, class3)
    sim_bc = calculate_path_similarity(class2, class3)

    if sim_ab is None or sim_ac is None or sim_bc is None:
        print("Error: Could not calculate similarity for all pairs. Returning None.")
        return None
    average_sim = (sim_ab + sim_ac + sim_bc) / 3
    
    return average_sim

def generate_and_score_permutations(input_file: str, output_file: str, num_permutations: int, seed: int):
    """
    Reads classes from a file, generates random permutations, calculates average
    similarity, and saves the results to an output CSV file with a header.

    Args:
        input_file: Path to the file with classes (e.g., 'imagenet_map.txt').
        output_file: Path to the output file to save results.
        num_permutations: The number of random permutations to generate.
        seed: The seed for the random number generator for reproducibility.
    """
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' was not found.")
        return
    model = load_model('large')
    random.seed(seed)

    class_names = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                class_names.append(parts[2])
    except Exception as e:
        print(f"Error reading from file: {e}")
        return

    if len(class_names) < 3:
        print("Error: Not enough classes in the file to create permutations of 3.")
        return

    print(f"Found {len(class_names)} classes. Generating {num_permutations} permutations...")

    results = []
    for i in range(num_permutations):
        sampled_classes = random.sample(class_names, 3)
        
        avg_score = calculate_average_similarity_of_three_classes(sampled_classes)
        score = measure_similarity(sampled_classes,model)

        if avg_score is not None and score is not None:
            results.append(sampled_classes + [avg_score]+[score])
            if (i + 1) % 100 == 0:print(f"Processed {i + 1}/{num_permutations} permutations.")
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class1', 'class2', 'class3', 'distance_score','semantic_score'])
            writer.writerows(results)
        print(f"\nSuccessfully wrote {len(results)} permutations to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to file: {e}")

def plot_similarity_histogram(input_file,bins):
    """
    Plots a histogram of the specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
    """
    dataframe = pd.read_csv(input_file)
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['semantic_score'], bins=bins, edgecolor='black')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('similarity_score_histogram.png')

def generate_training_instance(input_filename,output_filename,seed):
    np.random.seed(seed)
    df = pd.read_csv(input_filename)
    df['score'] = df['semantic_score'].round(4)
    df = df.drop('semantic_score',axis=1)

    df['seed'] = np.random.randint(0, 10000, size=len(df))
    df['early_epoch'] = np.random.randint(15, 25, size=len(df))
    df['max_epoch'] =  np.random.normal(loc=100, scale=15, size=len(df)).astype(int)
    df['store_weight'] = np.random.random(len(df)) < 0.3
    df['optimizer'] = np.where(np.random.random(len(df)) < 0.5, 'Adam', 'SGD')
    df['status'] = '-'

    # df['max_epoch'] =  1
    # df['early_epoch'] =  1
    df['store_weight'] =  True


    df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    generate_and_score_permutations('imagenet_map.txt', 'permutations.csv', 100, seed=42)
    plot_similarity_histogram('permutations.csv',100)
    generate_training_instance('permutations.csv','sweep.csv', seed=42)