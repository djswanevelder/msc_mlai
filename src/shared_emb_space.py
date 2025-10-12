import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from typing import Tuple, Dict, Any, Union
import optuna


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

def prepare_data_and_loaders(file_path: str, n_test: int, batch_size: int, num_loss_bins: int) -> Tuple[DataLoader, DataLoader, int, int, float, float]:
    """
    Loads the meta-dataset, performs loss binning, splits data into train/test sets,
    and creates PyTorch DataLoaders.

    Args:
        file_path (str): Path to the PyTorch meta-dataset file.
        n_test (int): Number of samples to reserve for the test set.
        batch_size (int): Batch size for the training DataLoader.
        num_loss_bins (int): Number of bins for loss quantization.

    Returns:
        Tuple[DataLoader, DataLoader, int, int, float, float]: 
            (train_dataloader, test_dataloader, weight_size, data_dim, min_loss, max_loss).
    """
    print(f"Loading data from {file_path}...")
    try:
        data = torch.load(file_path, weights_only=False)
        weights_tensor = data['weights'].float() 
        dataset_vector_tensor = torch.from_numpy(data['dataset_vector']).float()
        final_loss_tensor = torch.from_numpy(data['final_loss']).float()
    except Exception as e:
        print(f"Error loading data: {e}"); sys.exit(1)

    N = weights_tensor.shape[0]
    if N < n_test: 
        print(f"Error: Dataset size (N={N}) is too small to reserve {n_test} points for testing."); sys.exit(1)
        
    N_TRAIN = N - n_test

    weight_size = weights_tensor.shape[1]
    data_dim = dataset_vector_tensor.shape[1]
    
    # --- Loss Binning ---
    min_loss, max_loss = final_loss_tensor.min(), final_loss_tensor.max()
    # Create bins (excluding min/max to ensure indices are within [0, NUM_LOSS_BINS-1])
    bins = torch.linspace(min_loss, max_loss, num_loss_bins + 1)[1:-1]
    # Quantize the continuous loss values into discrete bins (indices)
    loss_indices = torch.clamp(torch.bucketize(final_loss_tensor, bins), 0, num_loss_bins - 1).long()

    # --- Data Splitting ---
    train_data, test_data = dataset_vector_tensor[:N_TRAIN], dataset_vector_tensor[-n_test:]
    train_loss_indices, test_loss_indices = loss_indices[:N_TRAIN], loss_indices[-n_test:]
    train_weights, test_weights = weights_tensor[:N_TRAIN], weights_tensor[-n_test:]

    # --- Setup DataLoaders ---
    train_dataset = TensorDataset(train_data, train_loss_indices, train_weights)
    # Drop last batch to ensure batch size is always even (required by NT-Xent)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    # Test set loader (one single batch of size N_TEST)
    test_dataset = TensorDataset(test_data, test_loss_indices, test_weights)
    test_dataloader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)
    
    return train_dataloader, test_dataloader, weight_size, data_dim, min_loss.item(), max_loss.item()

def predict_latent_vector(
    model_path: str, 
    dataset_embedding: torch.Tensor, 
    validation_loss: float
) -> torch.Tensor:
    """
    Predicts the weight latent vector for a given dataset embedding and validation loss,
    loading the trained model and its configuration from a file path.

    Args:
        model_path (str): File path to the trained encoder model containing weights and config.
        dataset_embedding (torch.Tensor): The 1536D dataset vector (shape [1536] or [1, 1536]).
        validation_loss (float): The continuous validation loss achieved by the run.

    Returns:
        torch.Tensor: The predicted weight latent vector (shape [1, WEIGHT_SIZE]).
    """
    
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
    """
    Main function to load the dataset, train the WeightLatentEncoder using NT-Xent 
    loss, save the model, and evaluate it on a held-out test set.
    
    Args:
        config (Dict): Dictionary containing all necessary file paths and hyperparameters.
    """
    
    # Unpack configuration
    file_path = config['DATASET_FILE_PATH']
    model_path = config['OUTPUT_MODEL_PATH']
    n_test = config['N_TEST']
    batch_size = config['BATCH_SIZE']
    num_epochs = config['NUM_EPOCHS']
    temp = config['TEMPERATURE']
    loss_embedding_k = config['LOSS_EMBEDDING_K']
    num_loss_bins = config['NUM_LOSS_BINS']
    
    # --- 1. Load and Prepare Data ---
    train_loader, test_loader, weight_size, data_dim, min_loss, max_loss = prepare_data_and_loaders(
        file_path, n_test, batch_size, num_loss_bins
    )
    
    N_TRAIN = len(train_loader.dataset)
    
    # --- 2. Model Setup and Training Loop ---
    encoder = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    print("-" * 50)
    print(f"Starting Training | N_Train: {N_TRAIN} | N_Test: {n_test} | Weight Dim: {weight_size}")
    print(f"Loss Range for Binning: [{min_loss:.4f}, {max_loss:.4f}]")
    print("-" * 50)

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        total_loss = 0
        
        for batch_data, batch_loss_indices, batch_weights in train_loader:
            optimizer.zero_grad()
            
            # Predict the weight latent vector
            predicted_weights = encoder(batch_data, batch_loss_indices)
            
            # Concatenate predictions (Anchor) and ground truths (Positive)
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            
            # Calculate contrastive loss
            loss = calculate_nt_xent_loss(all_embeddings, temp)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Log training progress
        print(f"Epoch {epoch}/{num_epochs} | Avg Training Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            print("-" * 50)

    # --- 3. Save Trained Model Weights and Configuration ---
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
    print(f"\nTraining complete. Encoder weights and config saved to: {model_path}")

    # --- 4. Load and Evaluate (Demonstrate Inference) ---
    print("\n" + "=" * 50)
    print(f"Loading saved weights and running prediction on test set.")
    
    # Re-initialize and load the model (using the saved file path)
    predictor = WeightLatentEncoder(data_dim, num_loss_bins, loss_embedding_k, weight_size)
    predictor.load_state_dict(save_data['state_dict']) # Use the in-memory save_data for quick load
    predictor.eval() # Set to evaluation mode

    print("=" * 50)

    # Evaluation on Excluded Test Data
    with torch.no_grad():
        for batch_data, batch_loss_indices, batch_weights in test_loader:
            # Forward Pass (Prediction)
            predicted_weights = predictor(batch_data, batch_loss_indices)
            # NT-Xent Loss
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            test_loss_ntxent = calculate_nt_xent_loss(all_embeddings, temp)
            
            # MSE Loss (for interpretability/distance)
            test_loss_mse = F.mse_loss(predicted_weights, batch_weights)

            print(f"Test Batch Loss (NT-Xent): {test_loss_ntxent.item():.4f}")
            print(f"Test Batch Loss (MSE Distance): {test_loss_mse.item():.6f}")

    print("\nProcess complete.")
    return test_loss_mse

# -------------------------------------------------------------
#                  OPTUNA OBJECTIVE FUNCTION
# -------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """
    Defines the search space for Optuna and calls the training pipeline.
    """
    
    # --- 1. Define Search Space ---
    
    # Learning Rate: Log-uniform sampling is best for LR
    lr = trial.suggest_float('LEARNING_RATE', 1e-5, 1e-3, log=True)
    
    # Batch Size: Integer sampling, ensuring it's an even number (for NT-Xent)
    # 10 is the absolute minimum, 100 or 128 is a good maximum for a 200 sample set.
    batch_size = trial.suggest_int('BATCH_SIZE', 10, 60, step=2) 
    
    # Loss Embedding Dimension (k)
    loss_embedding_k = trial.suggest_int('LOSS_EMBEDDING_K', 64, 256, step=32)
    
    # Number of Bins (must be >= 2)
    num_loss_bins = trial.suggest_int('NUM_LOSS_BINS', 10, 80, step=10)
    
    # Temperature for NT-Xent (can also be tuned)
    temp = trial.suggest_float('TEMPERATURE', 0.05, 0.5, log=True)
    
    # Other fixed parameters
    config = {
        'DATASET_FILE_PATH': 'data/final_meta_dataset.pt',
        # NOTE: Not saving model weights during tuning to speed up
        'OUTPUT_MODEL_PATH': f'data/trials/temp_trial_{trial.number}.pth', 
        'N_TEST': 10,
        'NUM_EPOCHS': 50, # Reduce epochs for tuning speed
        
        # Sampled parameters
        'LEARNING_RATE': lr,
        'BATCH_SIZE': batch_size,
        'LOSS_EMBEDDING_K': loss_embedding_k,
        'NUM_LOSS_BINS': num_loss_bins,
        'TEMPERATURE': temp,
    }
    
    # --- 2. Run Training and Return Metric ---
    
    # The run_training_pipeline function is modified to return the test_loss_mse
    mse_loss = run_training_pipeline(config)
    
    return mse_loss


# if __name__ == "__main__":
    
#     # --- OPTUNA STUDY SETUP ---
    
#     # 1. Create a study object
#     # direction='minimize' tells Optuna to find the parameters that result in the smallest return value.
#     study = optuna.create_study(
#         direction='minimize', 
#         sampler=optuna.samplers.TPESampler(seed=42) # TPE is the Bayesian approach
#     )
    
#     # 2. Run the optimization
#     N_TRIALS = 100 # Set a reasonable number of trials
#     print(f"Starting Optuna search for {N_TRIALS} trials...")
    
#     # The objective function is called N_TRIALS times, with Optuna intelligently
#     # suggesting new hyperparameters in each call.
#     study.optimize(objective, n_trials=N_TRIALS)

#     # 3. Report Results
#     print("\n" + "=" * 60)
#     print("✨ Optimization Finished ✨")
#     print("-" * 60)
#     print(f"Number of finished trials: {len(study.trials)}")
#     print(f"Best Trial Number: {study.best_trial.number}")
#     print(f"Best Test MSE Loss: {study.best_value:.6f}")
#     print("\nBest hyperparameters found:")
#     for key, value in study.best_params.items():
#         print(f"  {key}: {value}")
#     print("=" * 60)

# BATCH_SIZE: 52
# LOSS_EMBEDDING_K: 64
# NUM_LOSS_BINS: 70
# TEMPERATURE: 0.19010694880943896


if __name__ == "__main__":

    CONFIG = {
    'DATASET_FILE_PATH': 'data/final_meta_dataset.pt',
    'OUTPUT_MODEL_PATH': 'data/trained_encoder_weights.pth',
    # Dimensions and Binning
    'LOSS_EMBEDDING_K': 64,
    'NUM_LOSS_BINS': 50,
    # Training Hyperparameters
    'TEMPERATURE': 0.19,
    'BATCH_SIZE': 52,
    'NUM_EPOCHS': 200,
    'N_TEST': 10
    }


    test_loss_mse = run_training_pipeline(CONFIG)