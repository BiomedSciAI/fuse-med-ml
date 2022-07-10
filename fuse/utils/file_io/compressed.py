import zipfile
from tqdm import tqdm
import os


def extract_zip_file(filename: str, dest_dir: str, show_progress_bar=True):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"extracting {filename} ...")
    with zipfile.ZipFile(filename) as zf:
        if show_progress_bar:
            for member in tqdm(zf.infolist(), desc="Extracting "):
                zf.extract_zip_file(member, dest_dir)
        else:
            zf.extractall(dest_dir)
