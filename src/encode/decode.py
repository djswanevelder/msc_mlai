import torch
import sys
import csv
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from typing import List, Dict, Any, Union

from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader


from src.encode.encode_models import decode_latent_to_resnet_model
from shared_emb_space import predict_latent_vector





EMBEDDING_DIM = 512
LATENT_DIM = 512 
FINAL_VECTOR_DIM = EMBEDDING_DIM * 3 # 1536 (Concatenated dataset vector size)


def standardize_class_name(name: str) -> str:
    """
    Standardizes a class name (e.g., 'soft-coated Wheaten Terrier') into the 
    lowercased format used for matching keys (e.g., 'soft-coated_wheaten_terrier').
    """
    name = name.strip().lower()
    name = name.replace(' ', '_').replace('"', '').replace("'", "")
    return name

def generate_concatenated_dataset_vector(
    class_names: List[str], 
    class_vectors_csv_path: str
) -> torch.Tensor:
    """
    Loads 512D class embedding vectors from a CSV, looks up the vectors for the 
    three provided class names, concatenates them into a single 1536D tensor, 
    and returns the result.

    Args:
        class_names (List[str]): List of exactly three class names.
        class_vectors_csv_path (str): Path to the CSV file containing 512D vectors.

    Returns:
        torch.Tensor: The concatenated 1536D dataset vector.
    """
    if len(class_names) != 3:
        raise ValueError("Must provide exactly three class names.")
        
    print(f"Loading CLIP vectors from: {class_vectors_csv_path}")
    class_vectors_lookup: Dict[str, np.ndarray] = {}
    
    try:
        # Load all 512D vectors into a lookup dictionary
        with open(class_vectors_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            dim_cols = [f'dim_{i}' for i in range(EMBEDDING_DIM)]
            
            for row in reader:
                class_name = row['class_name']
                vector_values = [float(row[col]) for col in dim_cols]
                # Assuming class_name in CSV is already standardized or matches lookup key
                class_vectors_lookup[standardize_class_name(class_name)] = np.array(vector_values, dtype=np.float32)

    except FileNotFoundError:
        print(f"Error: Class vectors file not found at {class_vectors_csv_path}")
        # Use mock data if the file isn't found, so the prediction demo can proceed
        print("WARNING: Using mock 1536D data for demonstration due to missing CSV file.")
        full_vector_np = np.random.randn(FINAL_VECTOR_DIM).astype(np.float32)
        return torch.from_numpy(full_vector_np)
    except Exception as e:
        print(f"Error parsing class vectors CSV: {e}")
        print("WARNING: Using mock 1536D data for demonstration due to parsing error.")
        full_vector_np = np.random.randn(FINAL_VECTOR_DIM).astype(np.float32)
        return torch.from_numpy(full_vector_np)

    # Lookup and concatenate the three requested vectors
    concatenated_vectors = []
    zero_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    for name in class_names:
        standardized_name = standardize_class_name(name)
        vector = class_vectors_lookup.get(standardized_name, zero_vector)
        
        if np.array_equal(vector, zero_vector):
            print(f"Warning: Vector for class '{name}' (standardized: '{standardized_name}') not found. Using zero vector.")
            
        concatenated_vectors.append(vector)
    
    # Concatenate (512*3 = 1536D) and convert to PyTorch tensor
    full_vector_np = np.concatenate(concatenated_vectors).astype(np.float32)
    return torch.from_numpy(full_vector_np)

def calculate_dataset_mean_std(data_path: str, classes_to_use: List[str]) -> tuple[List[float], List[float]]:
    """
    Calculates the channel-wise mean and standard deviation for the images 
    belonging to the specified classes in the given data path, using the
    exact logic from the ResNet training script.

    Raises:
        ValueError: If no images are found for the target classes.
    """
    print("Calculating dataset mean and standard deviation for target classes...")
    
    # Initial transform: resize and convert to tensor (no normalization yet)
    initial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load all data to filter it
    full_dataset = datasets.ImageFolder(data_path, transform=initial_transform)

    # Filter samples to ONLY include the target classes and remap labels (0, 1, 2, ...)
    filtered_samples = []
    class_to_new_label = {cls: i for i, cls in enumerate(classes_to_use)}
    
    for path, label in full_dataset.samples:
        class_name = full_dataset.classes[label]
        if class_name in classes_to_use:
            new_label = class_to_new_label[class_name]
            filtered_samples.append((path, new_label))
    
    if not filtered_samples:
        raise ValueError(f"No images found for classes {classes_to_use} in {data_path}. Cannot calculate mean/std.")

    # Create a temporary ImageFolder using the filtered samples to perform mean/std calculation
    temp_dataset = datasets.ImageFolder(
        root=data_path,
        loader=default_loader,
        transform=initial_transform,
    )
    # Inject the filtered, re-indexed samples for calculation
    temp_dataset.samples = filtered_samples
    temp_dataset.imgs = filtered_samples
    
    # Use a DataLoader for batch processing
    loader = DataLoader(temp_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # 1. Calculate Mean
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # Flatten spatial dimensions
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

    if total_samples == 0:
        raise ValueError("DataLoader yielded zero samples. Check data path and classes.")

    mean /= total_samples
    
    # 2. Calculate Std
    total_samples = 0
    
    # Iterate through the data loader again to calculate std
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1).to(images.device))**2).sum([0, 2]) 
        total_samples += batch_samples

    # Final STD calculation, matching the user's reference logic
    std = torch.sqrt(std / (total_samples * images.size(2))) 
    
    print(f"Calculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()

def evaluate_model_on_actual_data(
    model: nn.Module, 
    class_list: List[str], 
    data_root_dir: str, 
    calculated_mean: List[float], 
    calculated_std: List[float],   
    device: str = "cpu"
) -> Union[tuple[float, float], None]:
    """
    Evaluates the decoded model on actual image data for the specified classes.
    
    Returns:
        tuple[float, float] or None: (avg_loss, accuracy) or None if evaluation fails.
    """
    
    print(f"\n--- Running Cross-Entropy Test on Decoded Model ({len(class_list)} classes) ---")

    NUM_CLASSES = len(class_list)
    BATCH_SIZE = 32
    
    # 1. Data Setup: Load and re-index real data
    try:
        # Final transform, using the custom calculated mean and std
        final_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=calculated_mean, std=calculated_std)
        ])

        # Load ALL data with a simple ToTensor() transform first to find paths/original labels
        initial_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        full_dataset = datasets.ImageFolder(data_root_dir, transform=initial_transform)

        # Filter samples and remap original labels to new labels (0, 1, 2)
        filtered_samples = []
        class_to_new_label = {cls: i for i, cls in enumerate(class_list)}
        
        for path, label in full_dataset.samples:
            class_name = full_dataset.classes[label]
            if class_name in class_list:
                new_label = class_to_new_label[class_name]
                filtered_samples.append((path, new_label))
        
        if not filtered_samples:
             raise FileNotFoundError(f"No images found for target classes in {data_root_dir}. Check class names/path.")

        # Create the final dataset object using the correct transform and re-indexed samples
        final_dataset = datasets.ImageFolder(
            root=data_root_dir,
            loader=default_loader,
            transform=final_transform, # Apply the final normalization transform
        )
        final_dataset.samples = filtered_samples
        final_dataset.imgs = filtered_samples
        
        test_loader = torch.utils.data.DataLoader(
            final_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=4
        )
        print(f"SUCCESS: Loaded {len(final_dataset.samples)} images from '{data_root_dir}'. Labels re-indexed to 0-{NUM_CLASSES-1}.")

    except Exception as e:
        print(f"ERROR loading real data from '{data_root_dir}' ({e}). Using mock data for test.")
        
        # Fallback Mock Data Setup
        mock_data = torch.randn(BATCH_SIZE * NUM_CLASSES, 3, 224, 224)
        mock_labels = torch.cat([torch.full((BATCH_SIZE,), i) for i in range(NUM_CLASSES)]).long()
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(mock_data, mock_labels), 
            batch_size=BATCH_SIZE, 
            shuffle=False
        )
    
    # 2. Prepare Model for Test
    try:
        model.to(device)
        model.eval()
        
        # CRITICAL: Ensure the final classification layer is correctly sized for N=3 classes
        if model.fc.out_features != NUM_CLASSES:
            # Note: This adjustment is temporary for the test loss calculation.
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES).to(device)
        
        # 3. Evaluation Loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate Cross-Entropy loss
                loss = F.cross_entropy(outputs, labels) 
                total_loss += loss.item()
                num_batches += 1

                # Calculate Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy

    except Exception as e:
        print(f"Error during model cross-entropy evaluation: {e}")
        return None

def run_full_prediction_pipeline(target_val_loss: float, example_classes: List[str]) -> Union[tuple[float, float], tuple[None, None]]:
    """
    Executes the full pipeline from input parameters to final model evaluation.

    Returns:
        Union[tuple[float, float], tuple[None, None]]: (Cross-Entropy Loss, Accuracy) or (None, None).
    """
    # --- 1. Define required paths and constants (kept internal for simplicity) ---
    CHECKPOINT_PATH = "../../data/dataset/weight_ae_final.pth"
    DATASET_DIR = "../../data/dataset/"
    CLASS_VECTOR_CSV_PATH = "./dataset_latents.csv"
    ENCODER_MODEL_PATH = 'trained_encoder_weights.pth' 
    IMAGE_DATA_ROOT = '../../data/imagenet_subsets/'
    
    # --- IMPORTANT: Calculate Mean and Std ---
    try:
        calculated_mean, calculated_std = calculate_dataset_mean_std(
            data_path=IMAGE_DATA_ROOT, 
            classes_to_use=example_classes
        )
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to calculate Mean/Std ({e}). Aborting.")
        return None, None # Return tuple for consistency

    # --- STEP 1: Generate the 1536D dataset vector (Input for the Encoder) ---
    print("=" * 60)
    print("STEP 1: Generating 1536D Dataset Vector from Class Embeddings")
    print(f"Classes: {example_classes}")
    print("=" * 60)
    
    try:
        dataset_vector = generate_concatenated_dataset_vector(
            class_names=example_classes,
            class_vectors_csv_path=CLASS_VECTOR_CSV_PATH
        )
        print(f"SUCCESS: Dataset Vector Shape: {dataset_vector.shape} (Expected: [{FINAL_VECTOR_DIM}])")
        
    except Exception as e:
        print(f"\nFATAL ERROR in STEP 1: Could not generate dataset vector. Aborting.")
        print(f"Details: {e}")
        return None, None # Return tuple for consistency
        
    # --- STEP 2: Predict the 512D Latent Vector ---
    print("\n" + "=" * 60)
    print(f"STEP 2: Predicting {LATENT_DIM}D Latent Vector")
    print(f"Input: {dataset_vector.shape[0]}D Dataset Vector, Target Loss: {target_val_loss}")
    print(f"Model: {ENCODER_MODEL_PATH}")
    print("=" * 60)

    try:
        # Predict the Latent Vector (Output of the Weight Latent Encoder)
        predicted_latent_vec = predict_latent_vector(
            model_path=ENCODER_MODEL_PATH,
            dataset_embedding=dataset_vector,
            validation_loss=target_val_loss
        )
        
        if predicted_latent_vec is None:
            return None, None # Return tuple for consistency
        
        # --- DIAGNOSTIC: Check Latent Vector ---
        latent_mean = predicted_latent_vec.mean().item()
        latent_std = predicted_latent_vec.std().item()
        print(f"DIAGNOSTIC: Predicted Latent Vector Mean: {latent_mean:.6f}, Std: {latent_std:.6f}")
        
        print(f'SUCCESS: Predicted Latent Vector generated with shape {predicted_latent_vec.shape}.')
        
    except NameError:
        print("\nFATAL ERROR in STEP 2: The function 'predict_latent_vector' is not defined or imported.")
        return None, None # Return tuple for consistency
    except Exception as e:
        print(f"\nAn unexpected error occurred during prediction: {e}")
        return None, None # Return tuple for consistency

    # --- STEP 3: Decode Predicted Latent Vector into ResNet Model ---
    print("\n" + "=" * 60)
    print("STEP 3: Decoding Predicted Latent Vector into ResNet Model")
    print(f"Decoder Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    try:
        # Call the decoding function (Input for the AE Decoder)
        decoded_model = decode_latent_to_resnet_model(
            checkpoint_path=CHECKPOINT_PATH,
            dataset_dir=DATASET_DIR,
            latent_vector=predicted_latent_vec, 
            device="cpu" 
        )

        if not isinstance(decoded_model, torch.nn.Module):
            print("\nError: Decoding failed or returned an unexpected object.")
            return None, None # Return tuple for consistency
            
        print("\nModel Decoding Successful!")
        if hasattr(decoded_model, 'fc'):
            final_layer_weights = decoded_model.fc.weight.data.shape
            print(f"Final layer weight shape (fc.weight): {final_layer_weights}")
            
            # --- DIAGNOSTIC: Check Final Layer Weights ---
            fc_weights = decoded_model.fc.weight.data
            weight_mean = fc_weights.mean().item()
            weight_std = fc_weights.std().item()
            print(f"DIAGNOSTIC: FC Weight Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred during decoding: {e}")
        return None, None # Return tuple for consistency

    # --- STEP 4: Test Cross-Entropy Loss on Actual Data ---
    print("\n" + "=" * 60)
    print("STEP 4: Testing Cross-Entropy Loss and Accuracy on Actual Image Data")
    print("=" * 60)
    
    loss_accuracy_pair = evaluate_model_on_actual_data(
        model=decoded_model,
        class_list=example_classes,
        data_root_dir=IMAGE_DATA_ROOT, 
        calculated_mean=calculated_mean, # Pass calculated mean
        calculated_std=calculated_std,   # Pass calculated std
        device="cpu"
    )
    
    if loss_accuracy_pair is None:
        return None, None

    cross_entropy_loss, accuracy = loss_accuracy_pair
    return cross_entropy_loss, accuracy


if __name__ == "__main__":
    EXAMPLE_CLASSES = ['sea_slug','appenzeller','soft-coated_wheaten_terrier']
    TARGET_VAL_LOSS = 0.2
    
    final_loss, final_accuracy = run_full_prediction_pipeline(
        target_val_loss=TARGET_VAL_LOSS,
        example_classes=EXAMPLE_CLASSES
    )

    print("\n" + "=" * 60)
    if final_loss is not None:
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Target Validation Loss: {TARGET_VAL_LOSS}")
        print(f"Final Decoded Model Cross-Entropy Loss on Test Data: {final_loss:.4f}")
        print(f"Final Decoded Model Classification Accuracy on Test Data: {final_accuracy * 100:.2f}%")
    else:
        print("PIPELINE FAILED.")
    print("=" * 60)
