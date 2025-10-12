import torch
import numpy as np
import os
import csv
import sys
from typing import Dict, List, Any

# --- Static Configuration ---
EMBEDDING_DIM = 512
FINAL_VECTOR_DIM = EMBEDDING_DIM * 3 # 1536 (512D per class * 3 classes)

def standardize_class_name(name: str) -> str:
    """
    Standardizes a class name (e.g., 'soft-coated Wheaten Terrier') into the 
    lowercased format used for matching against keys in the class vectors dictionary 
    (e.g., 'soft-coated_wheaten_terrier').
    
    Args:
        name (str): The class name string.

    Returns:
        str: The standardized name.
    """
    name = name.strip().lower()
    name = name.replace(' ', '_').replace('"', '').replace("'", "")
    return name

def load_clip_class_vectors(class_vectors_csv_path: str) -> Dict[str, np.ndarray]:
    """
    Loads the pre-computed 512D CLIP class embedding vectors from the CSV file 
    into a dictionary, keyed by standardized class name, for fast lookup.
    
    Args:
        class_vectors_csv_path (str): Path to the CSV file containing class embeddings.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping standardized class names to their 
                               512-dimensional NumPy array vectors.
    """
    print(f"Loading CLIP class vectors from: {class_vectors_csv_path}")
    class_vectors_lookup = {}
    
    try:
        with open(class_vectors_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Find the names of the dimension columns (dim_0 to dim_511)
            dim_cols = [f'dim_{i}' for i in range(EMBEDDING_DIM)]
            
            for row in reader:
                class_name = row['class_name']
                # Extract vector values and convert to numpy array (float32)
                vector_values = [float(row[col]) for col in dim_cols]
                class_vectors_lookup[class_name] = np.array(vector_values, dtype=np.float32)
                
        if not class_vectors_lookup:
            print("Error: Class vector CSV is empty.")
            sys.exit(1)
            
        print(f"Successfully loaded {len(class_vectors_lookup)} unique class vectors.")
        return class_vectors_lookup

    except FileNotFoundError:
        print(f"Error: Class vectors file not found at {class_vectors_csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing class vectors CSV: {e}")
        sys.exit(1)

def load_artifact_metadata(metadata_csv_path: str) -> List[Dict]:
    """
    Loads the run-level metadata (containing class names, artifact_id, and val_loss) 
    from the input CSV file.
    
    Args:
        metadata_csv_path (str): Path to the CSV file containing artifact and run details.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a row/run.
    """
    print(f"Loading artifact metadata from: {metadata_csv_path}")
    try:
        with open(metadata_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            metadata_list = list(reader)
            print(f"Loaded {len(metadata_list)} metadata entries.")
            return metadata_list
    except FileNotFoundError:
        print(f"Error: Artifact metadata file not found at {metadata_csv_path}")
        sys.exit(1)

def assemble_meta_dataset(
    weights: torch.Tensor,
    run_metadata: List[Dict],
    class_vectors_lookup: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Combines the weight latent vectors, concatenated class vectors, and validation 
    loss values into the final meta-dataset dictionary structure.
    
    The input 'weights' tensor and 'run_metadata' list must be perfectly aligned 
    (same order and number of entries).
    
    Args:
        weights (torch.Tensor): Tensor of weight latent vectors, shape (N, D_weights).
        run_metadata (List[Dict]): List of run metadata, aligned with the weights tensor.
        class_vectors_lookup (Dict[str, np.ndarray]): Lookup table for CLIP class vectors.

    Returns:
        Dict[str, Any]: The final meta-dataset dictionary ready for saving.
    """
    print("Aggregating final meta-dataset components...")
    
    N = len(weights) 
    dataset_vectors = []
    loss_values = []
    
    for i in range(N):
        run = run_metadata[i]
        
        # --- 1. Get standardized class names ---
        c1_name = standardize_class_name(run.get('class1', ''))
        c2_name = standardize_class_name(run.get('class2', ''))
        c3_name = standardize_class_name(run.get('class3', ''))
        
        # --- 2. Look up 512D vectors ---
        zero_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1 = class_vectors_lookup.get(c1_name, zero_vector)
        v2 = class_vectors_lookup.get(c2_name, zero_vector)
        v3 = class_vectors_lookup.get(c3_name, zero_vector)
        
        # --- 3. Concatenate (1536D) ---
        full_vector = np.concatenate([v1, v2, v3]).astype(np.float32)
        dataset_vectors.append(full_vector)
        
        # --- 4. Append Loss ---
        loss = 0.0
        try:
            # Assuming 'val_loss' is the loss we want to predict
            loss = float(run.get('val_loss', 0.0))
        except (ValueError, TypeError):
            print(f"Warning: Invalid 'val_loss' for run {run.get('run_id')}. Setting to 0.0.")
            
        loss_values.append(loss)

    # Convert lists to final NumPy formats
    final_dataset_vectors = np.stack(dataset_vectors, axis=0) # Shape: (N, 1536)
    final_loss_values = np.array(loss_values, dtype=np.float32) # Shape: (N,)
    
    # --- 5. Construct final dataset dictionary ---
    final_dataset = {
        'weights': weights,                     # PyTorch Tensor (N, D_weights)
        'dataset_vector': final_dataset_vectors,  # NumPy Array (N, 1536)
        'final_loss': final_loss_values         # NumPy Array (N,)
    }
    
    return final_dataset


def run_meta_dataset_pipeline(class_vectors_csv: str, metadata_csv: str, weights_file: str, output_pt_path: str):
    """
    Orchestrates the loading, matching, and aggregation pipeline to create the 
    final meta-dataset file.
    
    Args:
        class_vectors_csv (str): Path to the CLIP class vectors CSV.
        metadata_csv (str): Path to the artifact/run metadata CSV (e.g., weights.csv).
        weights_file (str): Path to the PyTorch file containing the encoded weights.
        output_pt_path (str): Path to save the final PyTorch meta-dataset file.
    """
    
    # --- 1. Load pre-computed class embeddings ---
    class_vectors_lookup = load_clip_class_vectors(class_vectors_csv)
    
    # --- 2. Load run metadata (class names, loss, and key for matching) ---
    run_metadata_raw = load_artifact_metadata(metadata_csv)
    
    # --- 3. Load weight latent vectors ---
    try:
        data = torch.load(weights_file, map_location='cpu')
        # Assuming the dictionary of interest is keyed by 'encoded_models'
        weights_lookup_dict = data.get('encoded_models', {})
        if not weights_lookup_dict:
            # Fallback in case the entire file IS the dictionary
            weights_lookup_dict = data if isinstance(data, dict) else {}
        
        if not weights_lookup_dict:
             print(f"Error: Could not find encoded models in '{weights_file}'.")
             sys.exit(1)
             
    except FileNotFoundError:
        print(f"Error: Encoded weights file not found at {weights_file}")
        sys.exit(1)
    
    # --- 4. Order and Filter: Match Metadata to Weights ---
    N_metadata_requested = len(run_metadata_raw)
    ordered_weights_list = []
    runs_to_include = [] 
    
    print(f"Attempting to match {len(weights_lookup_dict)} weight latents to {N_metadata_requested} metadata entries...")

    for i, run in enumerate(run_metadata_raw):
        # The key in the weights dictionary is assumed to be in the 'artifact_name' column
        weight_key = run.get('artifact_name')
        
        if not weight_key:
            continue

        # Check for the key exactly as it appears in the CSV (e.g., UUID.pth or UUID)
        weight_tensor = weights_lookup_dict.get(weight_key)
        
        # Check for the key without the .pth suffix if the first check fails
        if weight_tensor is None and weight_key.endswith('.pth'):
             key_no_suffix = weight_key[:-4]
             weight_tensor = weights_lookup_dict.get(key_no_suffix)

        if weight_tensor is None:
            # print(f"Warning: Weight tensor not found for key '{weight_key}'. Skipping metadata row {i}.")
            continue

        if not isinstance(weight_tensor, torch.Tensor):
            print(f"Error: Found key '{weight_key}', but value is a {type(weight_tensor)}, not a torch.Tensor. Skipping.")
            continue
            
        ordered_weights_list.append(weight_tensor)
        runs_to_include.append(run)

    # Finalize the weights tensor and the metadata list
    if not ordered_weights_list:
        print(f"Fatal Error: Found 0 weight tensors corresponding to the {N_metadata_requested} metadata entries.")
        sys.exit(1)
        
    weights = torch.stack(ordered_weights_list, dim=0)
    run_metadata = runs_to_include
    N_final = len(weights)
    
    if N_final < N_metadata_requested:
        print(f"Warning: Only {N_final} runs successfully matched and will be processed (out of {N_metadata_requested} requested).")

    # --- 5. Aggregate and construct final dataset ---
    final_dataset = assemble_meta_dataset(weights, run_metadata, class_vectors_lookup)
    
    # --- 6. Save final dataset ---
    torch.save(final_dataset, output_pt_path)
    print(f"\nSuccessfully saved final meta-dataset (N={len(weights)}) to {output_pt_path}")
    print(f"Weights Tensor Shape: {final_dataset['weights'].shape}")
    print(f"Dataset Vector Shape: {final_dataset['dataset_vector'].shape}")
    print(f"Final Loss Shape: {final_dataset['final_loss'].shape}")


if __name__ == "__main__":
    
    # --- Configuration Variables (Replacing Terminal Inputs) ---
    CLASS_VECTORS_CSV = "data/dataset_latents.csv"  # Output from the previous script (512D vectors)
    ARTIFACT_METADATA_CSV = "data/meta_dataset_info.csv"    # Output from the run processing script (contains artifact_name, class1-3, loss)
    ENCODED_WEIGHTS_FILE = "data/model_latents.pt" # Input file containing the latent vectors of the models
    OUTPUT_PYTORCH_FILE = "data/final_meta_dataset.pt" # The final aggregated dataset file

    # 1. Run the entire aggregation pipeline
    run_meta_dataset_pipeline(
        class_vectors_csv=CLASS_VECTORS_CSV,
        metadata_csv=ARTIFACT_METADATA_CSV,
        weights_file=ENCODED_WEIGHTS_FILE,
        output_pt_path=OUTPUT_PYTORCH_FILE
    )
    
    # 2. Example of loading and inspecting the final saved file (kept from user's snippet)
    print("\n--- Inspecting the saved file ---")
    data = torch.load(OUTPUT_PYTORCH_FILE, weights_only=False)
    
    if len(data['final_loss']) > 0:
        i = 0
        loss = data['final_loss'][i]
        dataset_vector = data['dataset_vector'][i]
        weights_vector = data['weights'][i]
        
        print(f"First entry loss: {loss:.4f}")
        print(f"First entry dataset vector shape: {dataset_vector.shape}")
        print(f"First entry weights vector shape: {weights_vector.shape}")
    else:
        print("Dataset is empty.")
