
import hydra
from omegaconf import DictConfig
import requests
import os
import tarfile
import sys


def load_mapping(filepath: str)->dict:
    class_to_id = {} #{'class_name':{"class.id"}}
    with open(filepath,'r') as f:
        for line in f:
            out = line.strip().split()
            class_to_id[out[2].lower()] = out[0]
    return class_to_id

def has_images(target_path: str) -> bool:
    """Checks if a directory contains .JPEG files."""
    return any(fname.endswith(".JPEG") for fname in os.listdir(target_path))

def has_tar(target_path: str) -> str | None:
    """Returns the path to a .tar file if it exists, otherwise None."""
    for fname in os.listdir(target_path):
        if fname.endswith(".tar"):
            return os.path.join(target_path, fname)
    return None

def is_tar_complete(tar_path: str, url: str) -> bool:
    """Checks if a tar file is fully downloaded based on content size."""
    try:
        response = requests.head(url)
        expected_size = int(response.headers.get('content-length', 0))
        return os.path.getsize(tar_path) == expected_size
    except Exception:
        return False

def download_and_extract(source_url: str, target_path: str) -> None:
    """Downloads, extracts, and deletes the tar file."""
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
                downloaded_size += len(chunk)
                f.write(chunk)
                progress = (downloaded_size / total_size_in_bytes) * 100 if total_size_in_bytes > 0 else 0
                sys.stdout.write(f"\r  - Progress: {progress:.2f}%")
                sys.stdout.flush()
        
        sys.stdout.write('\n')
    except requests.exceptions.RequestException as e:
            print(f"  - Error downloading {source_url}: {e}")
    
    print(f"Extracting {tar_file_name}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_path)
    
    print(f"Deleting {tar_file_name}...")
    os.remove(tar_path)
    print("Process completed.")

@hydra.main(version_base=None, config_name="dataset.yaml", config_path="conf/")
def main(cfg: DictConfig, dataset_path: str = "imagenet_subsets") -> None:
    class_to_id = load_mapping('imagenet_map.txt')
    os.makedirs(dataset_path, exist_ok=True)

    for c in cfg.classes:
        target_path = os.path.join(dataset_path, c)
        source_url = f'https://image-net.org/data/winter21_whole/{class_to_id[c]}.tar'

        if not os.path.exists(target_path):
            print(f"Folder for {c} not found. Creating and downloading.")
            download_and_extract(source_url, target_path)
            continue
            
        tar_path = has_tar(target_path)
        images_exist = has_images(target_path)
        
        # Determine state and action
        if images_exist:
            if tar_path:
                print(f"Images and incomplete tar found for {c}. Deleting tar.")
                os.remove(tar_path)
            print(f"Download for {c} already complete.")
            continue
        
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
        else: # No tar file, no images
            print(f"No files found for {c}. Downloading and extracting.")
            download_and_extract(source_url, target_path)

if __name__ == "__main__":
    main()