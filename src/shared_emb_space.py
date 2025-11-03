import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from typing import Tuple, Dict, Any, Union, List
import optuna
import matplotlib.pyplot as plt
import wandb
import os 
from pathlib import Path # Added for robust file path handling

# Dimensions (Default values, dynamically set during data loading)
WEIGHT_SIZE = 512       
DATA_DIM = 1536         

class WeightLatentEncoder(nn.Module):
    """
    A neural network that attempts to map a meta-dataset description (concatenated 
    CLIP vectors) combined with a binned loss index to the corresponding 
    weight latent vector.
    
    The architecture uses a simple fully connected network.
    """
    def __init__(self, data_dim: int, num_bins: int, embedding_k: int, output_size: int):
        """
        Initializes the Encoder network.
        
        Args:
            data_dim (int): Dimensionality of the meta-dataset vector (e.g., 1536).
            num_bins (int): Number of bins used for discretizing the final loss.
            embedding_k (int): Dimensionality of the loss embedding (e.g., 128).
            output_size (int): Dimensionality of the output weight latent vector (e.g., 512).
        """
        super().__init__()
        # Loss index is embedded
        self.loss_embedder = nn.Embedding(num_embeddings=num_bins, embedding_dim=embedding_k)
        
        input_size = data_dim + embedding_k
        
        # Simple feed-forward network
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size) # Output size matches the weight latent size
        )

    def forward(self, data: torch.Tensor, loss_index: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass by concatenating the data and the loss embedding.
        
        Args:
            data (torch.Tensor): The dataset vector (e.g., 1536D CLIP vectors).
            loss_index (torch.Tensor): The binned index representing the run's final loss.

        Returns:
            torch.Tensor: The predicted weight latent vector.
        """
        loss_emb = self.loss_embedder(loss_index)
        x = torch.cat([data, loss_emb], dim=-1)
        return self.net(x)

def calculate_nt_xent_loss(embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates the Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss.
    
    This function implements contrastive loss for paired comparison, assuming the 
    input `embeddings` is concatenated as [Predictions (Anchor); Ground Truths (Positive)].
    
    Args:
        embeddings (torch.Tensor): Concatenated tensor [Predictions; Ground Truths], 
                                   shape (2 * BATCH_SIZE, D).
        temperature (float): Scaling factor for the similarity calculation.

    Returns:
        torch.Tensor: The scalar NT-Xent loss.
    """
    
    norm_embeddings = F.normalize(embeddings, dim=1)
    
    similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
    
    mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    similarity_matrix = similarity_matrix / temperature
    
    half_batch_size = embeddings.size(0) // 2 
    indices_1 = torch.arange(half_batch_size, device=embeddings.device)
    indices_2 = torch.arange(half_batch_size, device=embeddings.device) + half_batch_size
    
    log_prob = F.log_softmax(similarity_matrix, dim=1)
    
    log_prob_1 = log_prob[indices_1, indices_2]
    log_prob_2 = log_prob[indices_2, indices_1]
    
    loss = -(log_prob_1.mean() + log_prob_2.mean()) / 2
    return loss

def prepare_data_and_loaders(file_path: str, n_val: int, n_test: int, batch_size: int, num_loss_bins: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, float, float, torch.Tensor, torch.Tensor]:
    """
    Loads the meta-dataset, performs loss binning, splits data into train/validation/test sets,
    and creates PyTorch DataLoaders.

    Args:
        file_path (str): Path to the PyTorch meta-dataset file.
        n_val (int): Number of samples to reserve for the validation set.
        n_test (int): Number of samples to reserve for the test set.
        batch_size (int): Batch size for the DataLoaders.
        num_loss_bins (int): Number of bins for loss quantization.

    Returns:
        Tuple: (train_dataloader, val_dataloader, test_dataloader, weight_size, data_dim, min_loss, max_loss, 
                final_loss_tensor, loss_indices)
    """
    print(f"Loading data from {file_path}...")
    try:
        data = torch.load(file_path, weights_only=False) 
        weights_tensor = data['weights'].float() 
        dataset_vector_tensor = torch.from_numpy(data['dataset_vector']).float()
        final_loss_tensor = torch.from_numpy(data['final_loss']).float()
        # NOTE: file_paths are needed for custom sample selection in the heatmap plotting
        if 'file_paths' not in data:
             # Create dummy paths if not present, but warn user
             print("Warning: 'file_paths' not found in dataset. Heatmap customization will fail.")
             file_paths = [f"sample_{i}.pth" for i in range(weights_tensor.shape[0])]
        else:
             file_paths = data['file_paths']
    except Exception as e:
        print(f"Error loading data: {e}"); sys.exit(1)

    N = weights_tensor.shape[0]
    n_reserved = n_val + n_test
    if N < n_reserved: 
        print(f"Error: Dataset size (N={N}) is too small to reserve {n_reserved} points for validation and testing."); sys.exit(1)
        
    N_TRAIN = N - n_reserved

    weight_size = weights_tensor.shape[1]
    data_dim = dataset_vector_tensor.shape[1]
    
    # --- Loss Binning ---
    min_loss, max_loss = final_loss_tensor.min(), final_loss_tensor.max()
    # Create bins (excluding min/max to ensure indices are within [0, NUM_LOSS_BINS-1])
    bins = torch.linspace(min_loss, max_loss, num_loss_bins + 1)[1:-1]
    # Quantize the continuous loss values into discrete bins (indices)
    loss_indices = torch.clamp(torch.bucketize(final_loss_tensor, bins), 0, num_loss_bins - 1).long()

    # --- Data Splitting (Train | Val | Test) ---
    N_VAL_START = N_TRAIN
    N_VAL_END = N_TRAIN + n_val
    N_TEST_START = N_VAL_END

    # Train set (0 to N_TRAIN)
    train_data, train_loss_indices, train_weights = (
        dataset_vector_tensor[:N_TRAIN], loss_indices[:N_TRAIN], weights_tensor[:N_TRAIN]
    )
    
    # Validation set (N_TRAIN to N_TRAIN + N_VAL)
    val_data, val_loss_indices, val_weights = (
        dataset_vector_tensor[N_VAL_START:N_VAL_END], loss_indices[N_VAL_START:N_VAL_END], weights_tensor[N_VAL_START:N_VAL_END]
    )
    
    # Test set (N_TRAIN + N_VAL to N)
    test_data, test_loss_indices, test_weights = (
        dataset_vector_tensor[N_TEST_START:], loss_indices[N_TEST_START:], weights_tensor[N_TEST_START:]
    )

    # --- Setup DataLoaders ---
    train_dataset = TensorDataset(train_data, train_loss_indices, train_weights)
    # Drop last batch to ensure batch size is always even (required by NT-Xent)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    val_dataset = TensorDataset(val_data, val_loss_indices, val_weights)
    # Validation loader does not shuffle
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Test set loader (one single batch of size N_TEST)
    test_dataset = TensorDataset(test_data, test_loss_indices, test_weights)
    test_dataloader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)
    
    # Returning the file_paths array as well
    return train_dataloader, val_dataloader, test_dataloader, weight_size, data_dim, min_loss.item(), max_loss.item(), final_loss_tensor, loss_indices, file_paths

def run_validation_epoch(encoder: WeightLatentEncoder, val_loader: DataLoader, temp: float) -> float:
    """Calculates the average NT-Xent loss on the validation set."""
    encoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, batch_loss_indices, batch_weights in val_loader:
            # NT-Xent requires an even batch size for paired comparison
            if batch_data.size(0) % 2 != 0:
                continue 
            
            predicted_weights = encoder(batch_data, batch_loss_indices)
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            loss = calculate_nt_xent_loss(all_embeddings, temp)
            total_loss += loss.item()
            
    # Calculate average loss, considering only processed batches
    num_processed_batches = sum(1 for batch in val_loader if batch[0].size(0) % 2 == 0)
    
    return total_loss / num_processed_batches if num_processed_batches > 0 else 0.0

def predict_latent_vector(
    model_path: str, 
    dataset_embedding: torch.Tensor, 
    validation_loss: float
) -> torch.Tensor:
    # 1. Load model state and configuration
    try:
        loaded_data = torch.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

    # Extract dimensions and binning configuration
    min_loss = loaded_data['min_loss']
    max_loss = loaded_data['max_loss']
    num_loss_bins = loaded_data['num_loss_bins']
    weight_size = loaded_data['weight_size']
    data_dim = loaded_data['data_dim']
    loss_embedding_k = loaded_data['loss_embedding_k']

    # 2. Initialize the model
    model = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    model.load_state_dict(loaded_data['state_dict'])
    model.eval()

    # 3. Convert single loss float to binned index (replicating training logic)
    with torch.no_grad():
        # Ensure the input loss is a tensor
        loss_tensor = torch.tensor([validation_loss], dtype=torch.float)

        # Create the same bins used during training
        min_loss_t = torch.tensor(min_loss)
        max_loss_t = torch.tensor(max_loss)
        bins = torch.linspace(min_loss_t, max_loss_t, num_loss_bins + 1)[1:-1]
        
        # Quantize the continuous loss value into a discrete bin index
        # Clamp ensures the index stays within [0, NUM_LOSS_BINS - 1]
        loss_index = torch.clamp(torch.bucketize(loss_tensor, bins), 0, num_loss_bins - 1).long()
        
        # 4. Ensure dataset embedding is correctly shaped (e.g., [1, 1536])
        if dataset_embedding.dim() == 1:
            dataset_embedding = dataset_embedding.unsqueeze(0)
            
        # 5. Perform the prediction
        predicted_latent = model(dataset_embedding, loss_index)
        
    return predicted_latent

def run_training_pipeline(config: Dict[str, Union[str, int, float]]):
  
    # 1. 🚀 Initialize WandB Run
    wandb.init(project="weight-latent-encoder", config=config)
    
    # Unpack configuration
    file_path = config['DATASET_FILE_PATH']
    model_path = config['OUTPUT_MODEL_PATH']
    n_test = config['N_TEST']
    n_val = config['N_VAL'] 
    batch_size = config['BATCH_SIZE']
    num_epochs = config['NUM_EPOCHS']
    temp = config['TEMPERATURE']
    loss_embedding_k = config['LOSS_EMBEDDING_K']
    num_loss_bins = config['NUM_LOSS_BINS']
    
    # --- 1. Load and Prepare Data ---
    # Note: We discard file_paths from the return here as they are not used for training
    train_loader, val_loader, test_loader, weight_size, data_dim, min_loss, max_loss, _, _, _ = prepare_data_and_loaders(
        file_path, n_val, n_test, batch_size, num_loss_bins
    )
    
    N_TRAIN = len(train_loader.dataset)
    N_VAL = len(val_loader.dataset)
    
    # --- 2. Model Setup and Training Loop ---
    encoder = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # 💡 Log model graph and watch parameters
    wandb.watch(encoder, log="all")

    print("-" * 50)
    print(f"Starting Training | N_Train: {N_TRAIN} | N_Val: {N_VAL} | N_Test: {n_test} | Weight Dim: {weight_size}")
    print(f"Loss Range for Binning: [{min_loss:.4f}, {max_loss:.4f}]")
    print("-" * 50)

    best_val_loss = float('inf')
    # Use the os.path.join for robust path construction
    best_model_save_path = model_path.replace(".pth", "_best_val.pth")

    for epoch in range(1, num_epochs + 1):
        # --- Training Epoch ---
        encoder.train()
        total_train_loss = 0
        
        for batch_data, batch_loss_indices, batch_weights in train_loader:
            optimizer.zero_grad()
            
            predicted_weights = encoder(batch_data, batch_loss_indices)
            
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            
            loss = calculate_nt_xent_loss(all_embeddings, temp)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Epoch ---
        avg_val_loss = run_validation_epoch(encoder, val_loader, temp)
        
        # 🚀 Log Training and Validation Loss to WandB
        wandb.log({
            "train/avg_nt_xent_loss": avg_train_loss, 
            "val/avg_nt_xent_loss": avg_val_loss,
            "epoch": epoch
        })
        
        # Log training progress to console
        print(f"Epoch {epoch}/{num_epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_data = {
                'state_dict': encoder.state_dict(),
                'min_loss': min_loss,
                'max_loss': max_loss,
                'num_loss_bins': num_loss_bins,
                'weight_size': weight_size,
                'data_dim': data_dim,
                'loss_embedding_k': loss_embedding_k
            }
            torch.save(save_data, best_model_save_path)
            print(f"   -> Model saved! New best validation loss: {best_val_loss:.4f}")
            
        if epoch % 10 == 0:
            print("-" * 50)

    # --- 3. Save Final Trained Model Weights and Configuration ---
    # Save the final state (useful if early stopping is not implemented)
    save_data = {
        'state_dict': encoder.state_dict(),
        'min_loss': min_loss,
        'max_loss': max_loss,
        'num_loss_bins': num_loss_bins,
        'weight_size': weight_size,
        'data_dim': data_dim,
        'loss_embedding_k': loss_embedding_k
    }
    torch.save(save_data, model_path)
    print(f"\nTraining complete. Final encoder weights and config saved to: {model_path}")


    # --- 4. Load and Evaluate (Demonstrate Inference on Test Set) ---
    print("\n" + "=" * 50)
    
    # Determine which model to use for testing: the best validation model or the final epoch model
    if os.path.exists(best_model_save_path) and N_VAL > 0 and best_val_loss != float('inf'):
        print(f"Loading best validation model from {best_model_save_path} for test.")
        final_model_to_test = best_model_save_path
    else:
        print(f"Using final epoch model from {model_path} for test.")
        final_model_to_test = model_path
        
    # Load the chosen model
    try:
        load_data = torch.load(final_model_to_test)
        predictor = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
        predictor.load_state_dict(load_data['state_dict']) 
        predictor.eval()
    except Exception as e:
        print(f"Error loading model for testing: {e}"); sys.exit(1)

    print("=" * 50)

    # Evaluation on Excluded Test Data
    with torch.no_grad():
        for batch_data, batch_loss_indices, batch_weights in test_loader:
            # Note: Test loader should always have an even batch size for NT-Xent
            if batch_data.size(0) % 2 != 0:
                print("Warning: Test batch size is odd. Cannot calculate NT-Xent loss. Skipping.")
                test_loss_ntxent_item = float('nan')
            else:
                predicted_weights = predictor(batch_data, batch_loss_indices)
                
                # NT-Xent Loss
                all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
                test_loss_ntxent = calculate_nt_xent_loss(all_embeddings, temp)
                test_loss_ntxent_item = test_loss_ntxent.item()
                
                # MSE Loss (for interpretability/distance)
                test_loss_mse = F.mse_loss(predicted_weights, batch_weights)
                test_loss_mse_item = test_loss_mse.item()

                # 🚀 Log Test Metrics to WandB (only done once since test_loader has one batch)
                wandb.log({
                    "test/nt_xent_loss": test_loss_ntxent_item,
                    "test/mse_distance": test_loss_mse_item
                })
                print(f"Test Batch Loss (NT-Xent): {test_loss_ntxent_item:.4f}")
                print(f"Test Batch Loss (MSE Distance): {test_loss_mse_item:.6f}")
                
            break 

    print("\nProcess complete.")
    
    wandb.finish()
    
    return test_loss_mse_item if 'test_loss_mse_item' in locals() else None

def evaluate_model_on_dataset(
    model_path: str, 
    data_file_path: str
) -> Tuple[float, float, int]:
    """
    Loads a trained encoder model and a meta-dataset file, computes the predicted 
    weight latents, and calculates the average Cosine Similarity and MSE distance 
    between predictions and ground truths.

    Args:
        model_path (str): Path to the saved .pth model file.
        data_file_path (str): Path to the meta-dataset .pt file to evaluate on.

    Returns:
        Tuple[float, float, int]: (avg_cosine_similarity, avg_mse_distance, num_samples)
    """
    print("-" * 50)
    print(f"Starting Evaluation on {data_file_path} using model: {model_path}")
    
    # 1. Load model configuration and state
    try:
        loaded_model_data = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}"); return float('nan'), float('nan'), 0
    
    min_loss = loaded_model_data['min_loss']
    max_loss = loaded_model_data['max_loss']
    num_loss_bins = loaded_model_data['num_loss_bins']
    weight_size = loaded_model_data['weight_size']
    data_dim = loaded_model_data['data_dim']
    loss_embedding_k = loaded_model_data['loss_embedding_k']

    # 2. Initialize and load the model
    encoder = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    encoder.load_state_dict(loaded_model_data['state_dict'])
    encoder.eval()
    
    # 3. Load the evaluation data
    try:
        data = torch.load(data_file_path, weights_only=False) 
        weights_tensor = data['weights'].float() 
        dataset_vector_tensor = torch.from_numpy(data['dataset_vector']).float()
        final_loss_tensor = torch.from_numpy(data['final_loss']).float()
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}"); return float('nan'), float('nan'), 0
    
    # 4. Replicate Loss Binning (Crucial for correct input)
    min_loss_t = torch.tensor(min_loss)
    max_loss_t = torch.tensor(max_loss)
    bins = torch.linspace(min_loss_t, max_loss_t, num_loss_bins + 1)[1:-1]
    loss_indices = torch.clamp(torch.bucketize(final_loss_tensor, bins), 0, num_loss_bins - 1).long()
    
    # 5. Inference and Metric Calculation
    num_samples = weights_tensor.shape[0]
    
    with torch.no_grad():
        # Predict the latent vectors
        predicted_weights = encoder(dataset_vector_tensor, loss_indices)
        
        # Calculate Cosine Similarity
        # Normalize the vectors
        pred_norm = F.normalize(predicted_weights, dim=1)
        gt_norm = F.normalize(weights_tensor, dim=1)
        # Cosine Similarity is the dot product of normalized vectors
        # Note: This computes the mean similarity across the entire dataset.
        cos_sim_values = (pred_norm * gt_norm).sum(dim=1)
        avg_cosine_similarity = cos_sim_values.mean().item()
        
        # Calculate Mean Squared Error (MSE)
        avg_mse_distance = F.mse_loss(predicted_weights, weights_tensor).item()

    print(f"Evaluation Complete (N={num_samples})")
    print(f"   -> Average Cosine Similarity: {avg_cosine_similarity:.4f} (Closer to 1.0 is better)")
    print(f"   -> Average MSE Distance: {avg_mse_distance:.6f} (Closer to 0.0 is better)")
    print("-" * 50)
    
    return avg_cosine_similarity, avg_mse_distance, num_samples


def plot_similarity_heatmap(model_path: str, data_file_path: str, target_identifiers: List[str]):
    """
    Generates and plots a Cosine Similarity distance matrix heatmap for a **specific subset**
    of the data, comparing ground truth weight embeddings (Rows) against predicted 
    embeddings (Columns).
    
    Rows = Ground Truth Weight Latents (W_i)
    Columns = Predicted Weight Latents (P_j)
    Values = Cosine Similarity(W_i, P_j)
    
    Args:
        model_path (str): Path to the trained encoder model.
        data_file_path (str): Path to the meta-dataset file.
        target_identifiers (List[str]): List of run file names (e.g., 'run_id_epoch.pth') to select.
    """
    print("-" * 50)
    print(f"Generating {len(target_identifiers)}x{len(target_identifiers)} Custom Similarity Heatmap.")
    
    # 1. Load model configuration and state
    try:
        loaded_model_data = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot generate heatmap."); return
    
    min_loss = loaded_model_data['min_loss']
    max_loss = loaded_model_data['max_loss']
    num_loss_bins = loaded_model_data['num_loss_bins']
    weight_size = loaded_model_data['weight_size']
    data_dim = loaded_model_data['data_dim']
    loss_embedding_k = loaded_model_data['loss_embedding_k']

    # 2. Initialize and load the model
    encoder = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    encoder.load_state_dict(loaded_model_data['state_dict'])
    encoder.eval()
    
    # 3. Load the evaluation data and file paths
    try:
        data = torch.load(data_file_path, weights_only=False) 
        weights_tensor = data['weights'].float() 
        dataset_vector_tensor = torch.from_numpy(data['dataset_vector']).float()
        final_loss_tensor = torch.from_numpy(data['final_loss']).float()
        
        if 'file_paths' not in data:
            raise KeyError("Dataset file must contain 'file_paths' key for custom selection.")
            
        dataset_file_paths = data['file_paths']
        # Sanitize file paths: extract just the filename if full paths are stored
        dataset_file_names = [Path(p).name for p in dataset_file_paths]

    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}"); return
    
    # 4. Find the indices matching the target identifiers
    target_indices = []
    found_identifiers = []
    
    # Create a set for quick lookup
    target_set = set(target_identifiers)
    
    for i, file_name in enumerate(dataset_file_names):
        if file_name in target_set:
            target_indices.append(i)
            found_identifiers.append(file_name)
    
    if len(target_indices) != len(target_identifiers):
        missing = target_set - set(found_identifiers)
        print(f"Warning: Found {len(target_indices)}/{len(target_identifiers)} target samples. Missing: {missing}")
        if not target_indices:
             print("Aborting heatmap generation due to no matching samples.")
             return
    
    # Ensure the order of selected data matches the order of requested identifiers
    # This involves re-indexing the slices according to the order of found_identifiers
    sorted_indices = []
    # Map found identifiers back to their indices in the dataset, ensuring the order matches the input list
    name_to_index = {name: index for index, name in zip(target_indices, found_identifiers)}
    
    for identifier in target_identifiers:
        if identifier in name_to_index:
            sorted_indices.append(name_to_index[identifier])
    
    # The actual samples used and their labels (must be in the same order)
    subset_indices = torch.tensor(sorted_indices, dtype=torch.long)
    final_labels = [target_identifiers[i] for i in range(len(target_identifiers)) if target_identifiers[i] in found_identifiers]

    # 5. Extract Subset Data
    subset_weights = weights_tensor[subset_indices]        # Ground Truth (Rows)
    subset_data = dataset_vector_tensor[subset_indices]    # Dataset Vector for Prediction
    
    # Replicate Loss Binning (Crucial for correct input)
    min_loss_t = torch.tensor(min_loss)
    max_loss_t = torch.tensor(max_loss)
    bins = torch.linspace(min_loss_t, max_loss_t, num_loss_bins + 1)[1:-1]
    loss_indices = torch.clamp(torch.bucketize(final_loss_tensor, bins), 0, num_loss_bins - 1).long()
    subset_loss_indices = loss_indices[subset_indices] # Loss Index for Prediction
    
    # 6. Prediction
    with torch.no_grad():
        predicted_weights = encoder(subset_data, subset_loss_indices) # Predicted (Columns)

    # 7. Calculate Cosine Similarity Matrix (NxN)
    # Normalize vectors
    predicted_norm = F.normalize(predicted_weights, dim=1)
    gt_norm = F.normalize(subset_weights, dim=1)
    
    # Calculate the similarity matrix: (Rows: GT, Columns: Predicted)
    similarity_matrix = torch.matmul(gt_norm, predicted_norm.T).cpu().numpy()
    
    N_plot = len(final_labels)
# 8. Plot Heatmap
    N_plot = len(final_labels)
    
    # 1. Enable LaTeX for plotting - IMPORTANT
    # This should be at the start of your script or before the first plot call
    plt.rcParams['text.usetex'] = True 
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # --- Custom Label Generation ---
    # Assume we use the provided structure for the labels
    example_target_labels = [
        "Dataset-Result 1 (10)", "Dataset-Result 1 (65)",
        "Dataset-Result 2 (14)", "Dataset-Result 2 (41)", 
        "Dataset-Result 3 (18)", "Dataset-Result 3 (42)"
    ]

    x_labels = []
    y_labels = []

    for label_str in example_target_labels[:N_plot]:
        dataset_num = label_str.split(' ')[1] 
        aux_val = label_str.split('(')[1].strip(')') 
        
        latex_label = rf'$\substack{{\left(\mathcal{{D}},\mathcal{{R}}\right)_{{{dataset_num}}} \\ ({aux_val})}}$'
        x_labels.append(latex_label)
        y_label = rf'$\substack{{\mathbf{{W}}_{{{dataset_num}}} \\ ({aux_val})}}$'
        y_labels.append(y_label)

    # -----------------------------

    plt.figure(figsize=(12, 11))
    # plt.imshow(similarity_matrix, cmap='viridis', aspect='equal', vmin=-0.7, vmax=0.7)
    img = plt.imshow(similarity_matrix, cmap='coolwarm', aspect='equal', vmin=-0.5, vmax=0.5)
    plt.colorbar(
        img, 
        label=r'Cosine Similarity $(\mathbf{W}_i, \mathbf{P}_j)$',
        shrink=0.7,   
        aspect=20,
    )
    # plt.colorbar(label=r'Cosine Similarity $(\mathbf{W}_i, \mathbf{P}_j)$')
    
    # plt.title(r'Similarity Matrix: Ground Truth ($\mathbf{W}$) vs. Predicted ($\mathbf{P}$)', fontsize=14)
    plt.xlabel(r'Predicted Weight Vector', fontsize=18) # Title can be generic now
    plt.ylabel(r'Ground Truth Weight Vector', fontsize=18) # Title can be generic now
    
    # Apply the custom LaTeX labels
    plt.xticks(np.arange(N_plot), x_labels, rotation=0, fontsize=18) 
    plt.yticks(np.arange(N_plot), y_labels, fontsize=18) 
    
    # Add values to the cells
    # ... (rest of the plotting code remains the same)
    # Add values to the cells
    for i in range(N_plot):
        for j in range(N_plot):
            # Highlight diagonal (matching) entries
            color = 'gold' if i == j else 'black'
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                     ha='center', va='center', color=color, fontsize=14, 
                     fontweight='bold' if i == j else 'normal')
            
    plt.tight_layout()
    plt.show()
    print("-" * 50)


def plot_distribution_and_binned_distribution(losses: torch.Tensor, binned_losses: torch.Tensor, num_loss_bins: int):
    """
    Plots the distribution of continuous loss values and the binned distribution.

    Args:
        losses (torch.Tensor): The original continuous loss values.
        binned_losses (torch.Tensor): The binned (discrete index) loss values.
        num_loss_bins (int): The number of bins used for quantization.
    """
    
    # Convert tensors to numpy arrays for plotting
    losses_np = losses.numpy()
    binned_losses_np = binned_losses.numpy()

    # --- Plot the **Distribution of Continuous Losses** ---
    plt.figure(figsize=(10, 5))
    plt.hist(losses_np, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Losses (Continuous Values, N={len(losses_np)})")
    plt.xlabel("Loss Value (Original)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # --- Plot the **Binned Distribution of Losses** ---
    plt.figure(figsize=(10, 5))
    # Note: We use the actual number of bins and align bin centers for clarity
    plt.hist(binned_losses_np, bins=num_loss_bins, range=(-0.5, num_loss_bins - 0.5), color='lightcoral', edgecolor='black', rwidth=0.8)
    plt.title(f"Binned Distribution of Losses (Discrete Indices, K={num_loss_bins} Bins)")
    plt.xlabel("Bin Index")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, num_loss_bins, max(1, num_loss_bins // 10))) # Show ticks every 10% of bins or at least 1
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":

    CONFIG = {
    'DATASET_FILE_PATH': 'data/final_meta_dataset.pt',
    'OUTPUT_MODEL_PATH': 'data/trained_encoder_weights.pth',
    # Dimensions and Binning
    'LOSS_EMBEDDING_K': 128*2,
    'NUM_LOSS_BINS': 200,
    # Training Hyperparameters
    'TEMPERATURE': 0.19,
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 400,
    'N_VAL': 50,
    'N_TEST': 12 
    }
    

    TARGET_FILES = [
        '2eGS2DhfjNLMCg3u7xSBst_10.pth', '2eGS2DhfjNLMCg3u7xSBst_65.pth',
        '2H3mxCEkHjcFbNJLgG3Jvo_14.pth', '2H3mxCEkHjcFbNJLgG3Jvo_41.pth',
        '2HDhNdy6GEJvYy4ZjgVyiH_18.pth', '2HDhNdy6GEJvYy4ZjgVyiH_42.pth',
        # '2hUHiu8bKnc88GpH6tnLXj_18.pth', '2hUHiu8bKnc88GpH6tnLXj_56.pth',
        # '2jGPzZJrJNxU6ujivotyAi_10.pth', '2jGPzZJrJNxU6ujivotyAi_27.pth',
    ]
    
    # test_loss_mse = run_training_pipeline(CONFIG)

    best_model_path = CONFIG['OUTPUT_MODEL_PATH'].replace(".pth", "_best_val.pth")

    # evaluate_model_on_dataset(
    #     model_path=best_model_path, 
    #     data_file_path=CONFIG['DATASET_FILE_PATH']
    # )
    
    plot_similarity_heatmap(
        model_path=best_model_path, 
        data_file_path=CONFIG['DATASET_FILE_PATH'],
        target_identifiers=TARGET_FILES
    )