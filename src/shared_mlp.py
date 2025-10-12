import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import sys
from typing import Tuple, Dict, Any, Union
import optuna
import wandb 

# Dimensions (Default values, dynamically set during data loading)
WEIGHT_SIZE = 512       
DATA_DIM = 1536         

# =================================================================
#  Model and Loss Definitions (UNCHANGED)
# =================================================================

class WeightLatentEncoderContinuous(nn.Module):
    """
    A neural network that maps a meta-dataset vector combined with a 
    CONTINUOUS loss value to the corresponding weight latent vector.
    """
    def __init__(self, data_dim: int, loss_embedding_k: int, output_size: int):
        super().__init__()
        
        # MLP to embed the single continuous loss value into loss_embedding_k dimensions
        self.loss_mlp = nn.Sequential(
            nn.Linear(1, 2), 
            nn.ReLU(),
            nn.Linear(2, loss_embedding_k) 
        )
        
        input_size = data_dim + loss_embedding_k
        
        # Simple feed-forward network
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, data: torch.Tensor, continuous_loss: torch.Tensor) -> torch.Tensor:
        loss_input = continuous_loss.unsqueeze(-1)
        loss_emb = self.loss_mlp(loss_input) 
        x = torch.cat([data, loss_emb], dim=-1) 
        return self.net(x)

def calculate_nt_xent_loss(embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    """Calculates the Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss."""
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

# =================================================================
#  Data Preparation (MODIFIED to include validation set)
# =================================================================

def prepare_data_and_loaders(file_path: str, n_test: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, float, float]:
    """
    Loads the meta-dataset, splits data into train, validation, and test sets, 
    and creates PyTorch DataLoaders.
    
    Returns: (train_loader, val_loader, test_loader, ...)
    """
    VAL_RATIO = 0.1 # Use 10% of the training data for validation
    
    print(f"Loading data from {file_path}...")
    try:
        data = torch.load(file_path, weights_only=False)
        weights_tensor = data['weights'].float() 
        dataset_vector_tensor = torch.from_numpy(data['dataset_vector']).float()
        final_loss_tensor = torch.from_numpy(data['final_loss']).float() 
    except Exception as e:
        print(f"Error loading data: {e}"); sys.exit(1)

    N = weights_tensor.shape[0]
    if N < n_test + 10: # Ensure we have enough for test + at least a small validation set
        print(f"Error: Dataset size (N={N}) is too small to reserve {n_test} points for testing."); sys.exit(1)
        
    N_TOTAL_TRAIN = N - n_test
    N_VAL = int(N_TOTAL_TRAIN * VAL_RATIO)
    N_TRAIN = N_TOTAL_TRAIN - N_VAL
    
    if N_VAL < batch_size:
        print(f"Warning: Validation size ({N_VAL}) is less than batch size ({batch_size}). Using all {N_VAL} samples in one validation step.")
        
    weight_size = weights_tensor.shape[1]
    data_dim = dataset_vector_tensor.shape[1]
    
    min_loss, max_loss = final_loss_tensor.min().item(), final_loss_tensor.max().item()

    # --- Data Splitting ---
    full_train_data = TensorDataset(dataset_vector_tensor[:N_TOTAL_TRAIN], 
                                    final_loss_tensor[:N_TOTAL_TRAIN], 
                                    weights_tensor[:N_TOTAL_TRAIN])

    # Indices for train/validation split
    train_indices = list(range(N_TRAIN))
    val_indices = list(range(N_TRAIN, N_TOTAL_TRAIN))

    # Test set preparation (last N_TEST samples)
    test_data = TensorDataset(dataset_vector_tensor[-n_test:], 
                              final_loss_tensor[-n_test:], 
                              weights_tensor[-n_test:])

    # --- Setup DataLoaders ---
    train_subset = Subset(full_train_data, train_indices)
    val_subset = Subset(full_train_data, val_indices)
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    # FIX: drop_last=False for validation to avoid division by zero when batch_size > N_VAL
    # NOTE: If the last batch size is 1, NT-Xent will be 0.
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Test DataLoader: Must have drop_last=False to use all N_TEST samples. 
    # Must ensure N_TEST is even if multiple batches were used, but here N_TEST is the batch size.
    test_dataloader = DataLoader(test_data, batch_size=n_test, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, weight_size, data_dim, min_loss, max_loss 

# =================================================================
#  Training Pipeline (MODIFIED to return TEST NT-XENT loss)
# =================================================================

def run_training_pipeline(config: Dict[str, Union[str, int, float]], trial: optuna.Trial = None):
    """
    Trains the encoder, performs validation every 10 epochs, logs to wandb, 
    and returns the final test NT-Xent loss (the new objective).
    """
    
    # Unpack configuration
    file_path = config['DATASET_FILE_PATH']
    n_test = config['N_TEST']
    batch_size = config['BATCH_SIZE']
    num_epochs = config['NUM_EPOCHS']
    temp = config['TEMPERATURE']
    loss_embedding_k = config['LOSS_EMBEDDING_K']
    learning_rate = config['LEARNING_RATE'] 
    
    # 1. Load and Prepare Data
    try:
        train_loader, val_loader, test_loader, weight_size, data_dim, min_loss, max_loss = prepare_data_and_loaders(
            file_path, n_test, batch_size
        )
    except SystemExit:
        if wandb.run:
             wandb.finish(exit_code=1) 
        return float('inf') 
    
    N_TRAIN = len(train_loader.dataset)
    
    # 2. Model Setup and Training Loop
    encoder = WeightLatentEncoderContinuous(data_dim, loss_embedding_k, weight_size) 
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    print(f"Starting Trial | LR: {learning_rate:.1e} | Batch: {batch_size} | Loss_Emb_K: {loss_embedding_k}")

    for epoch in range(1, num_epochs + 1):
        # --- TRAINING STEP ---
        encoder.train()
        total_train_loss = 0
        
        for batch_data, batch_continuous_losses, batch_weights in train_loader: 
            optimizer.zero_grad()
            predicted_weights = encoder(batch_data, batch_continuous_losses) 
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            loss = calculate_nt_xent_loss(all_embeddings, temp)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        wandb.log({"train/nt_xent_loss_epoch": avg_train_loss}, step=epoch)
        
        # --- VALIDATION STEP (Every 10 epochs) ---
        if epoch % 10 == 0 or epoch == num_epochs:
            encoder.eval()
            total_val_ntxent_loss = 0
            total_val_mse_loss = 0
            
            # Check if val_loader is NOT empty before iterating and dividing
            if len(val_loader) > 0: 
                with torch.no_grad():
                    for batch_data, batch_continuous_losses, batch_weights in val_loader:
                        predicted_weights = encoder(batch_data, batch_continuous_losses)
                        
                        # 1. QUANTIFY COLLAPSE: Predicted Variance
                        average_predicted_variance = torch.mean(torch.var(predicted_weights, dim=0))
                        
                        # 2. QUANTIFY BASELINE: Target Variance (The Mean Predictor's MSE)
                        # We calculate the average variance across all features in the batch
                        average_target_variance = torch.mean(torch.var(batch_weights, dim=0))

                        print(f'Mean of predicted weights: {torch.mean(predicted_weights)}')
                        print(f'Var of predicted weights: {average_predicted_variance:.8f}') # Format for clarity
                        print(f'Var of target weights (MSE baseline): {average_target_variance:.8f}')

                        # 1. NT-Xent Validation Loss
                        all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
                        
                        # Only calculate NT-Xent if the batch size is >= 2
                        if all_embeddings.size(0) >= 2:
                            ntxent_loss = calculate_nt_xent_loss(all_embeddings, temp)
                            total_val_ntxent_loss += ntxent_loss.item()
                        
                        # 2. MSE Validation Loss (Always calculable if batch size >= 1)
                        mse_loss = F.mse_loss(predicted_weights, batch_weights)
                        total_val_mse_loss += mse_loss.item()
                        # print(torch.var(batch_weights))

                
                # Check for division by zero again, in case the only non-dropped batch had size 1
                if total_val_ntxent_loss > 0 or len(val_loader) > 0:
                    avg_val_ntxent_loss = total_val_ntxent_loss / len(val_loader)
                    avg_val_mse_loss = total_val_mse_loss / len(val_loader)
                else:
                    # Fallback for extremely small val sets where the final batch size is too small
                    avg_val_ntxent_loss = float('inf')
                    avg_val_mse_loss = float('inf')
                    
                # Log validation losses to wandb
                wandb.log({
                    "val/nt_xent_loss_epoch": avg_val_ntxent_loss, 
                    "val/mse_loss_epoch": avg_val_mse_loss
                }, step=epoch)
            else:
                avg_val_ntxent_loss = float('inf')
                avg_val_mse_loss = float('inf')
                print("WARNING: Validation loader is empty or only contains single-sample batches. Skipping validation loss calculation.")

            
            print(f"Epoch {epoch}/{num_epochs} | Train NT-Xent: {avg_train_loss:.4f} | Val NT-Xent: {avg_val_ntxent_loss:.4f} | Val MSE: {avg_val_mse_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{num_epochs} | Train NT-Xent: {avg_train_loss:.4f}")


    # 3. Final Evaluation on Excluded TEST Data
    encoder.eval()
    test_loss_ntxent = float('inf') # <--- Initialize the new return metric
    test_loss_mse = float('inf') 

    with torch.no_grad():
        for batch_data, batch_continuous_losses, batch_weights in test_loader:
            predicted_weights = encoder(batch_data, batch_continuous_losses) 
            
            # Calculate NT-Xent Loss on the test set
            all_embeddings = torch.cat([predicted_weights, batch_weights], dim=0)
            if all_embeddings.size(0) >= 2: # Ensure batch size is >= 2 for NT-Xent
                test_loss_ntxent = calculate_nt_xent_loss(all_embeddings, temp).item()
            else:
                print("FATAL: Test set batch size is less than 2. Cannot calculate NT-Xent loss.")
                test_loss_ntxent = float('inf') # Set to inf if calculation fails
                
            # Calculate MSE Loss on the test set
            test_loss_mse = F.mse_loss(predicted_weights, batch_weights).item()
            break 

    print(f"\nTrial Final Test NT-Xent Loss (Objective): {test_loss_ntxent:.4f}")
    print(f"Trial Final Test MSE Loss: {test_loss_mse:.6f}")
    
    # Log BOTH final metrics
    wandb.log({
        "test/nt_xent_loss_objective": test_loss_ntxent, 
        "test/mse_loss": test_loss_mse, 
        "final_epoch": num_epochs
    })
    
    # *** RETURN THE NEW OBJECTIVE: TEST NT-XENT LOSS ***
    return test_loss_ntxent

# =================================================================
#                  OPTUNA OBJECTIVE FUNCTION (MODIFIED)
# =================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Defines the search space for Optuna, initializes wandb, and calls the training pipeline.
    """
    
    # --- 1. Define Search Space ---
    lr = trial.suggest_float('LEARNING_RATE', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('BATCH_SIZE', 10, 60, step=2) 
    loss_embedding_k = trial.suggest_int('LOSS_EMBEDDING_K', 2, 32, step=2) 
    temp = trial.suggest_float('TEMPERATURE', 0.05, 0.5, log=True)
    num_epochs = trial.suggest_int('NUM_EPOCHS', 20, 150, step=10)

    # --- 2. Build Configuration ---
    config = {
        'DATASET_FILE_PATH': 'data/final_meta_dataset.pt',
        'OUTPUT_MODEL_PATH': f'data/trials/temp_trial_{trial.number}.pth', 
        'N_TEST': 10,
        'NUM_EPOCHS': num_epochs, 
        
        # Sampled parameters
        'LEARNING_RATE': lr,
        'BATCH_SIZE': 90,
        'LOSS_EMBEDDING_K': loss_embedding_k,
        'TEMPERATURE': temp,
    }

    # --- 3. Initialize wandb run ---
    wandb.init(
        project="weight-latent-encoder-tuning", 
        config=config, 
        name=f"trial-{trial.number}",
        reinit=True 
    )
    
    # --- 4. Run Training and Return Metric ---
    # The run_training_pipeline now returns the test_loss_ntxent
    ntxent_loss = run_training_pipeline(config, trial) 
    
    # --- 5. Finalize wandb run ---
    wandb.finish()
    
    # *** RETURN NT-XENT LOSS ***
    return ntxent_loss


if __name__ == "__main__":
    
    ## --- OPTUNA STUDY SETUP ---
    
    # # 1. Create a study object
    # study = optuna.create_study(
    #     direction='minimize', 
    #     sampler=optuna.samplers.TPESampler(seed=42) 
    # )
    
    # # 2. Run the optimization
    # N_TRIALS = 10 
    # print(f"Starting Optuna search for {N_TRIALS} trials...")
    
    # study.optimize(objective, n_trials=N_TRIALS) 

    # # 3. Report Results
    # print("\n" + "=" * 60)
    # print("✨ Optimization Finished ✨")
    # print("-" * 60)
    # print(f"Number of finished trials: {len(study.trials)}")
    # print(f"Best Trial Number: {study.best_trial.number}")
    # # Note: study.best_value is now the best NT-Xent loss
    # print(f"Best Test NT-Xent Loss: {study.best_value:.4f}")
    # print("\nBest hyperparameters found:")
    # for key, value in study.best_params.items():
    #     print(f"  {key}: {value}")
    # print("=" * 60)
    config = {
    'DATASET_FILE_PATH': 'data/final_meta_dataset.pt',
    'OUTPUT_MODEL_PATH': f'data/trials/temp_trial.pth', 
    'N_TEST': 1,
    'NUM_EPOCHS': 100, 
    # Sampled parameters
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 180,
    'LOSS_EMBEDDING_K': 125,
    'TEMPERATURE': 0.05,
    }

    wandb.init(
        project="weight-latent-encoder-tuning", 
        config=config, 
        name=f"test",
        reinit=True 
    )
    
    ntxent_loss = run_training_pipeline(config) 
    wandb.finish()


