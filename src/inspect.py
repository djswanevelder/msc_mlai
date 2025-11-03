import torch
from pathlib import Path

def inspect_data_indices(file_path: str):
    """
    Loads the PyTorch meta-dataset and prints the range of available array indices, 
    which are the only internal identifiers when 'file_paths' is missing.
    """
    print("-" * 50)
    print(f"Attempting to load: {file_path}")
    
    if not Path(file_path).exists():
        print(f"Error: File not found at path '{file_path}'.")
        return

    try:
        # Load the file with the necessary security bypass (weights_only=False)
        data = torch.load(file_path, weights_only=False) 
        
        # We rely on 'weights' for the size, as it's the core tensor
        weights_tensor = data['weights']
        N_samples = weights_tensor.shape[0]
        print(f'{weights_tensor}')
        print(f"Successfully loaded {N_samples} samples.")
        print(f"The 'weights' tensor has shape: {list(weights_tensor.shape)}")
        
        # The canonical array indices are 0 to N-1
        print("\nCanonical Array Indices (Internal ID) available in the dataset:")
        
        # Generate and print the first 50 indices
        indices = list(range(N_samples))
        
        print(f"Range: 0 to {N_samples - 1}")
        print(f"First 50 Indices: {indices[:50]}")
        
    except Exception as e:
        print(f"\nError: Failed to load file.")
        print(f"Details: {e}")
    
    print("-" * 50)


if __name__ == "__main__":
    # --- Configuration ---
    EXAMPLE_PTH_FILE = 'data/final_meta_dataset.pt'
    
    inspect_data_indices(EXAMPLE_PTH_FILE)
