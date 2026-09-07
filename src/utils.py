import os
import torch
import requests
from tqdm import tqdm

def download_sam_checkpoint(checkpoint_dir="models", checkpoint_name="sam_vit_b_01ec64.pth"):
    """
    Downloads the SAM model checkpoint if it does not already exist.
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_name}"
        print("Downloading SAM checkpoint...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(checkpoint_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download")
        else:
            print("SAM checkpoint downloaded successfully.")
    else:
        print("SAM checkpoint already exists.")

if __name__ == '__main__':
    download_sam_checkpoint()
