import hydra
from omegaconf import DictConfig, OmegaConf
import requests
import os
import tarfile
import sys
from typing import Dict, Optional

DATASET_PATH = os.path.join(os.getcwd(),'data','imagenet_data')


# --- Utility Functions ---

def load_mapping(filepath: str) -> Dict[str, str]:
    """
    Loads a mapping from class names to ImageNet ID strings from a text file.

    The expected file format is three columns separated by whitespace,
    where the first column is the ImageNet ID (e.g., 'n01440764') and the 
    third column is the class name (e.g., 'tench').

    Args:
        filepath: The path to the mapping file (e.g., 'imagenet_map.txt').

    Returns:
        A dictionary mapping lowercase class names to ImageNet IDs. 
        Example: {'class_name': 'n01440764'}
    """
    class_to_id = {}
    with open(filepath, 'r') as f:
        for line in f:
            out = line.strip().split()
            # The ID is in out[0], the class name is in out[2]
            if len(out) >= 3:
                class_to_id[out[2].lower()] = out[0]
    return class_to_id

def has_images(target_path: str) -> bool:
    """
    Checks if a directory contains any image files with the '.JPEG' extension.

    Args:
        target_path: The path to the directory to check.

    Returns:
        True if at least one '.JPEG' file is found, False otherwise.
    """
    if not os.path.exists(target_path):
        return False
    return any(fname.endswith(".JPEG") for fname in os.listdir(target_path))

def has_tar(target_path: str) -> Optional[str]:
    """
    Checks if a directory contains a tar file and returns its full path.

    Args:
        target_path: The path to the directory to check.

    Returns:
        The full path to the first found '.tar' file, or None if no tar file exists.
    """
    if not os.path.exists(target_path):
        return None
    for fname in os.listdir(target_path):
        if fname.endswith(".tar"):
            return os.path.join(target_path, fname)
    return None

def is_tar_complete(tar_path: str, url: str) -> bool:
    """
    Checks if a locally saved tar file has been completely downloaded 
    by comparing its file size with the expected size from the remote URL.

    Args:
        tar_path: The local path to the tar file.
        url: The source URL of the remote tar file.

    Returns:
        True if the local file size matches the 'content-length' header 
        from the URL, False otherwise (including errors).
    """
    try:
        response = requests.head(url)
        response.raise_for_status()
        expected_size = int(response.headers.get('content-length', 0))
        return os.path.getsize(tar_path) == expected_size
    except Exception:
        return False

def download_and_extract(source_url: str, target_path: str) -> None:
    """
    Downloads a file from a URL, extracts it (assuming it's a tar archive), 
    and then deletes the downloaded tar file to save disk space.

    Provides a terminal progress bar during the download phase.

    Args:
        source_url: The URL of the file to download (must be a .tar file).
        target_path: The directory where the file should be saved and extracted.
    """
    os.makedirs(target_path, exist_ok=True)
    tar_file_name = os.path.basename(source_url)
    tar_path = os.path.join(target_path, tar_file_name)
    
    print(f"Downloading {tar_file_name}...")
    try:
        response = requests.get(source_url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192

        downloaded_size = 0
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk: # filter out keep-alive new chunks
                    downloaded_size += len(chunk)
                    f.write(chunk)
                    # Progress bar calculation
                    progress = (downloaded_size / total_size_in_bytes) * 100 if total_size_in_bytes > 0 else 0
                    sys.stdout.write(f"\r  - Progress: {progress:.2f}%")
                    sys.stdout.flush()
        
        sys.stdout.write('\n')
    except requests.exceptions.RequestException as e:
            print(f"  - Error downloading {source_url}: {e}")
            return # Exit function on download failure
    
    # Extraction
    print(f"Extracting {tar_file_name}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=target_path)
    except tarfile.TarError as e:
        print(f"  - Error extracting {tar_file_name}: {e}. The file might be corrupted.")
        
    # Cleanup
    print(f"Deleting {tar_file_name}...")
    try:
        os.remove(tar_path)
    except OSError as e:
        print(f"  - Error deleting {tar_path}: {e}")
        
    print("Process completed.")

def download(cfg: DictConfig, dataset_path: str = DATASET_PATH) -> None:
    """
    The main entry point for downloading selected ImageNet classes using Hydra configuration.

    This function reads a list of target classes from the Hydra configuration object (cfg.classes)
    and attempts to download the corresponding tar archives, extract the images, and clean up.
    It performs a comprehensive check for existing files to handle scenarios like
    partially downloaded tars or already extracted images to ensure idempotency.

    Args:
        cfg: The Hydra configuration dictionary (containing the list of 'classes').
        dataset_path: The **exact absolute or relative path** to the directory where 
                      the ImageNet subsets should be saved.
    """
    # Use dataset_path directly as the full path to the dataset directory.
    dataset_path = DATASET_PATH
    full_dataset_path = os.path.abspath(dataset_path) 
    
    # Locate the mapping file relative to the script's execution environment.
    map_file_path = os.path.join(full_dataset_path, 'imagenet_map.txt') 
    class_to_id = load_mapping(map_file_path)
    
    os.makedirs(full_dataset_path, exist_ok=True)
    print(f"Target download directory set to: {full_dataset_path}")

    # Use the Hydra config list of classes
    for c in cfg.classes:
        # Check if class name is valid based on map
        if c not in class_to_id:
            print(f"Warning: Class '{c}' not found in mapping file. Skipping.")
            continue
            
        target_path = os.path.join(full_dataset_path, c)
        source_url = f'https://image-net.org/data/winter21_whole/{class_to_id[c]}.tar'

        print(f"\nProcessing configured class: {c}")

        # Check 1: Folder doesn't exist - download
        if not os.path.exists(target_path):
            print(f"Folder for {c} not found. Creating and downloading.")
            download_and_extract(source_url, target_path)
            continue
            
        tar_path = has_tar(target_path)
        images_exist = has_images(target_path)
        
        # Check 2: Images exist - complete (cleanup tar if necessary)
        if images_exist:
            if tar_path:
                print(f"Images and incomplete tar found for {c}. Deleting tar.")
                os.remove(tar_path)
            print(f"Download for {c} already complete.")
            continue
        
        # Check 3: Only tar exists (no images) - check completeness
        if tar_path:
            if is_tar_complete(tar_path, source_url):
                print(f"Complete tar file for {c} found. Extracting and deleting tar.")
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=target_path)
                os.remove(tar_path)
            else:
                print(f"Incomplete tar file for {c} found. Deleting and re-downloading.")
                os.remove(tar_path)
                download_and_extract(source_url, target_path)
        else: # Check 4: No tar file, no images (empty folder) - download
            print(f"No files found for {c}. Downloading and extracting.")
            download_and_extract(source_url, target_path)


if __name__ == "__main__":

    data_config = {"classes":['red_fox','dingo']}
    data_cfg = OmegaConf.create(data_config)
    download(data_cfg,dataset_path=DATASET_PATH)