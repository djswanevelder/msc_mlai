import torch
import torch.nn as nn
from PIL import Image
import os
import csv
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import open_clip

# --- Static Configuration ---
BATCH_SIZE = 32 # Batch size for GPU inference during embedding calculation
CLIP_MODEL_NAME = "ViT-B-32" 
EMBEDDING_DIM = 512 # Dimension of the CLIP embedding vector (fixed for ViT-B/32)

def standardize_folder_name(name: str) -> str:
    """
    Standardizes a human-readable class name (e.g., 'sea slug') into a format 
    expected for directory names (e.g., 'sea_slug').

    This ensures correct path lookup by converting to lowercase, replacing spaces 
    with underscores, and removing common punctuation like quotes and apostrophes.
    
    Args:
        name (str): The class name from the metadata CSV.

    Returns:
        str: The standardized, folder-compatible class name.
    """
    name = name.strip().lower()
    name = name.replace(' ', '_').replace('"', '').replace("'", "")
    return name

def calculate_mean_clip_embedding(
    class_name: str, 
    root_dir: str, 
    model: nn.Module, 
    preprocess: Any, 
    device: torch.device
) -> torch.Tensor:
    """
    Loads all images for a given class folder, computes their CLIP embeddings in batches,
    and returns the mean, normalized embedding vector for that class.
    
    Args:
        class_name (str): The standardized name of the subdirectory (e.g., 'sea_slug').
        root_dir (str): The base path where all class folders reside.
        model (nn.Module): The loaded CLIP model instance.
        preprocess (Any): The image preprocessing function provided by CLIP.
        device (torch.device): The device (CPU or GPU) to run inference on.

    Returns:
        torch.Tensor: A 1D tensor (EMBEDDING_DIM) representing the average, 
                      normalized CLIP embedding for the class. Returns a zero vector 
                      if the folder is not found or contains no images.
    """
    class_path = os.path.join(root_dir, class_name)
    
    if not os.path.isdir(class_path):
        print(f"Warning: Class directory not found for '{class_name}'. Skipping.")
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device) 

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No images found in '{class_name}'. Returning zero embedding.")
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)

    all_embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc=f"Embedding {class_name}"):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_images = []
        
        for filename in batch_files:
            try:
                img_path = os.path.join(class_path, filename)
                image = Image.open(img_path).convert("RGB")
                batch_images.append(preprocess(image))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue

        if batch_images:
            image_input = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                embeddings = model.encode_image(image_input).float()
                # Normalize individual embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings)

    if not all_embeddings:
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)

    # Concatenate all embeddings, compute the mean, and re-normalize the final vector
    full_embedding_tensor = torch.cat(all_embeddings, dim=0)
    mean_embedding = full_embedding_tensor.mean(dim=0)
    mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
    
    return mean_embedding

def extract_unique_standardized_classes(metadata_csv_path: str) -> List[str]:
    """
    Reads the input metadata CSV, extracts class names from 'class1', 'class2', 
    and 'class3' columns, standardizes them for folder lookup, and returns 
    a unique, sorted list of class names.
    
    Args:
        metadata_csv_path (str): Path to the CSV file containing the run metadata.

    Returns:
        List[str]: A sorted list of unique, standardized class names.
    """
    unique_classes = set()
    try:
        with open(metadata_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ['class1', 'class2', 'class3']:
                    if col in row and row[col]:
                        unique_classes.add(standardize_folder_name(row[col]))
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_csv_path}. Cannot extract classes.")
        sys.exit(1)
    
    return sorted(list(unique_classes))

def save_class_embeddings_to_csv(
    unique_classes: List[str],
    image_root_dir: str,
    output_csv_path: str,
    model: nn.Module, 
    preprocess: Any, 
    device: torch.device
):
    """
    Computes the mean CLIP vector for each unique class and saves the result 
    (class name and 512 dimensions) directly into a CSV file.
    
    Args:
        unique_classes (List[str]): List of standardized class names to process.
        image_root_dir (str): The root path to the image folders.
        output_csv_path (str): The path to save the resulting CSV.
        model (nn.Module): The loaded CLIP model.
        preprocess (Any): The image preprocessing function.
        device (torch.device): The device (CPU/GPU) for inference.
    """

    # CSV header: 'class_name' + 512 dimensions
    csv_header = ['class_name'] + [f'dim_{i}' for i in range(EMBEDDING_DIM)]
    
    with open(output_csv_path, 'w', newline='') as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=csv_header)
        csv_writer.writeheader()

        print(f"Starting to embed {len(unique_classes)} unique classes...")

        for i, class_name in enumerate(unique_classes):
            print(f"Processing Class {i+1}/{len(unique_classes)}: {class_name}")

            # Calculate the mean embedding vector
            mean_embedding = calculate_mean_clip_embedding(
                class_name, image_root_dir, model, preprocess, device
            ).cpu()

            # Prepare the row data dictionary
            row_data = {'class_name': class_name}
            
            # Map the 512 tensor dimensions to their respective column headers
            embedding_list = mean_embedding.tolist()
            for j in range(EMBEDDING_DIM):
                row_data[f'dim_{j}'] = embedding_list[j]

            csv_writer.writerow(row_data)
            outfile.flush()
        
        print(f"Successfully generated class vector CSV at '{output_csv_path}'.")

def run_data_embedding_pipeline(metadata_csv_path: str, image_root_dir: str, output_csv_path: str):
    """
    Executes the full pipeline: initializes the model, identifies classes, 
    and computes the class mean CLIP embeddings.
    
    Args:
        metadata_csv_path (str): Path to the input CSV containing class names.
        image_root_dir (str): Root directory for class image folders.
        output_csv_path (str): Path to save the output CSV with embeddings.
    """
    
    # --- 1. Setup Device and Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, 
            pretrained='openai',
            device=device
        )
        model.eval()
        print("CLIP model loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}. Ensure 'open_clip_torch' is installed.")
        sys.exit(1)

    # --- 2. Get Unique Class List ---
    print(f"Loading metadata and identifying unique classes...")
    unique_classes = extract_unique_standardized_classes(metadata_csv_path)
    print(f"Found {len(unique_classes)} unique classes to embed.")

    # --- 3. Generate and Save Class Vectors CSV ---
    save_class_embeddings_to_csv(
        unique_classes, 
        image_root_dir, 
        output_csv_path, 
        model, 
        preprocess, 
        device
    )


if __name__ == "__main__":
    METADATA_CSV = "meta-dataset_info.csv" # The CSV containing 'class1', 'class2', etc.
    IMAGE_ROOT = "../../data/imagenet_subsets" # The root folder where class image directories reside
    OUTPUT_VECTORS_CSV = "dataset_latents1.csv" # The file to save the final embeddings to

    run_data_embedding_pipeline(
        metadata_csv_path=METADATA_CSV,
        image_root_dir=IMAGE_ROOT,
        output_csv_path=OUTPUT_VECTORS_CSV
    )
