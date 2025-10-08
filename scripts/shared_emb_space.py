import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Model Definition: The Result Embedder ---

class ResultEmbedder(nn.Module):
    """
    Embeds the three components (Model, Dataset, Loss) of a training 
    experiment into a shared, low-dimensional space.
    
    The 'Results embedding table' is implemented here as three distinct
    linear layers, one for each input type, mapping it to the shared_dim.
    """
    def __init__(self, model_dim: int, data_dim: int, loss_dim: int = 2, shared_dim: int = 64):
        """
        Args:
            model_dim: Size N of the flattened NN model vector.
            data_dim: Size M of the dataset representation vector.
            loss_dim: Size 2 (train loss, validation loss).
            shared_dim: The dimension D of the final shared embedding space.
        """
        super().__init__()
        self.shared_dim = shared_dim
        
        # Linear layer for Model Weights (W_model)
        # This layer projects the N-sized model vector to the shared D-sized space.
        self.model_projector = nn.Sequential(
            nn.Linear(model_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
        # Linear layer for Dataset Vector (W_data)
        # This layer projects the M-sized dataset vector to the shared D-sized space.
        self.data_projector = nn.Sequential(
            nn.Linear(data_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
        # Linear layer for Loss Values (W_loss)
        # This layer projects the 2-sized loss vector to the shared D-sized space.
        self.loss_projector = nn.Sequential(
            nn.Linear(loss_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, shared_dim)
        )

    def forward(self, v_model: torch.Tensor, v_data: torch.Tensor, v_loss: torch.Tensor):
        """
        Projects the input vectors into the shared embedding space.
        
        Args:
            v_model: Batch of flattened model weight vectors (B, N).
            v_data: Batch of dataset representation vectors (B, M).
            v_loss: Batch of loss value vectors (B, 2).
            
        Returns:
            E_model, E_data, E_loss: Normalized embeddings (B, D).
        """
        E_model = self.model_projector(v_model)
        E_data = self.data_projector(v_data)
        E_loss = self.loss_projector(v_loss)
        
        # Normalize the embeddings to the unit sphere (essential for contrastive loss)
        E_model = F.normalize(E_model, dim=1)
        E_data = F.normalize(E_data, dim=1)
        E_loss = F.normalize(E_loss, dim=1)
        
        return E_model, E_data, E_loss

# --- 2. Contrastive Loss Function (NT-Xent Style) ---

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross-Entropy Loss.
    Used to enforce that positive pairs (e.g., E_model_i and E_data_i) have 
    high similarity, while negative pairs (E_model_i and E_data_j, j!=i) 
    have low similarity.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        Calculates NT-Xent loss between two sets of embeddings (anchor z_i, positive z_j).
        
        Args:
            z_i: Anchor embeddings (B, D).
            z_j: Positive embeddings (B, D).
            
        Returns:
            The computed NT-Xent loss value.
        """
        # 1. Compute similarity matrix (dot product of normalized vectors)
        # S[i, j] is the similarity between anchor i and positive j
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # 2. Identify positive pairs (the diagonal)
        # The positive pair for anchor i is positive i (index i)
        # The correct class for each row is the index of the positive sample, which is the diagonal
        labels = torch.arange(similarity_matrix.shape[0]).long().to(similarity_matrix.device)
        
        # 3. Compute Cross-Entropy Loss
        # For each anchor i, we want high probability (low cross-entropy) for label i
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

# --- 3. Example Usage and Training Loop Setup ---

def run_embedding_experiment():
    # Define hyper-parameters based on the user request
    BATCH_SIZE = 32
    N_MODEL_DIM = 512 # N-size vector (flattened model weights)
    M_DATA_DIM = 128  # M-size vector (dataset representation/features)
    LOSS_DIM = 2      # Train Loss, Validation Loss
    SHARED_DIM = 64   # D-size shared embedding space
    TEMPERATURE = 0.1
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    
    # 1. Initialize Model and Loss
    embedder = ResultEmbedder(N_MODEL_DIM, M_DATA_DIM, LOSS_DIM, SHARED_DIM)
    nt_xent = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(embedder.parameters(), lr=LEARNING_RATE)
    
    print(f"--- Contrastive Embedding Setup ---")
    print(f"Model Input (N): {N_MODEL_DIM}, Dataset Input (M): {M_DATA_DIM}, Shared Embedding (D): {SHARED_DIM}")
    print(f"Using NT-Xent Loss with Temperature: {TEMPERATURE}")

    for epoch in range(NUM_EPOCHS):
        # 2. Simulate Batch Data (B experiments)
        # Each row is one experiment: (Model_i, Data_i, Loss_i)
        v_model = torch.randn(BATCH_SIZE, N_MODEL_DIM)
        v_data = torch.randn(BATCH_SIZE, M_DATA_DIM)
        v_loss = torch.rand(BATCH_SIZE, LOSS_DIM) * 5 # Losses between 0 and 5
        
        # 3. Forward Pass
        E_model, E_data, E_loss = embedder(v_model, v_data, v_loss)
        
        # 4. Compute Total Contrastive Loss
        # We enforce two positive relationships:
        # A) Model_i is close to Data_i (Model-Data alignment)
        # B) Model_i is close to Loss_i (Model-Loss alignment)
        
        # Contrastive Loss A: E_model (Anchor) vs E_data (Positive)
        loss_model_data = nt_xent(E_model, E_data)
        
        # Contrastive Loss B: E_model (Anchor) vs E_loss (Positive)
        loss_model_loss = nt_xent(E_model, E_loss)
        
        # Total loss is the sum of the two alignment objectives
        total_loss = loss_model_data + loss_model_loss
        
        # 5. Backward Pass and Optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {total_loss.item():.4f} (M-D: {loss_model_data.item():.4f}, M-L: {loss_model_loss.item():.4f})")

    # 6. Post-Training Example: Checking Similarity
    print("\n--- Final Embedding Similarity Check ---")
    
    # Take the first experiment's embeddings
    E_model_0 = E_model[0].detach()
    E_data_0 = E_data[0].detach()
    E_loss_0 = E_loss[0].detach()
    
    # Take a random negative sample's embeddings (e.g., index 5)
    E_data_neg = E_data[5].detach()
    
    # Compute dot product similarity (higher is better)
    # The contrastive loss aims to make these positive similarities high
    sim_model_data_pos = torch.dot(E_model_0, E_data_0).item()
    sim_model_loss_pos = torch.dot(E_model_0, E_loss_0).item()
    
    # And negative similarities low
    sim_model_data_neg = torch.dot(E_model_0, E_data_neg).item()
    
    print(f"Positive Similarity (E_model_0 vs E_data_0): {sim_model_data_pos:.4f}")
    print(f"Positive Similarity (E_model_0 vs E_loss_0): {sim_model_loss_pos:.4f}")
    print(f"Negative Similarity (E_model_0 vs E_data_5): {sim_model_data_neg:.4f}")

if __name__ == "__main__":
    run_embedding_experiment()
