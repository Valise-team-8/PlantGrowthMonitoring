import os
import zipfile
import shutil
import random

ZIP_FILE = "Sunflower Compressed.zip"
DATASET_DIR = "dataset"
DAYS = ["day1", "day2", "day3"]

def extract_zip(zip_path, extract_to):
    print(f"[INFO] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[INFO] Extraction complete!")

def get_all_images(folder):
    # Recursively collect all images in subfolders
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))
    return images

def split_into_days(all_images, dest_folder=DATASET_DIR):
    os.makedirs(dest_folder, exist_ok=True)

    # Create day folders
    for day in DAYS:
        os.makedirs(os.path.join(dest_folder, day), exist_ok=True)

    random.shuffle(all_images)

    n = len(all_images) // 3
    splits = [all_images[:n], all_images[n:2*n], all_images[2*n:]]

    for day, imgs in zip(DAYS, splits):
        day_folder = os.path.join(dest_folder, day)
        for img_path in imgs:
            shutil.copy(img_path, os.path.join(day_folder, os.path.basename(img_path)))
    print(f"[INFO] Images split into {DAYS} successfully!")

def setup_dataset():
    tmp_extract_folder = "temp_sunflower"
    extract_zip(ZIP_FILE, tmp_extract_folder)

    all_images = get_all_images(tmp_extract_folder)
    print(f"[INFO] Found {len(all_images)} images in zip.")

    split_into_days(all_images, DATASET_DIR)

    shutil.rmtree(tmp_extract_folder)
    print("\n✅ Dataset setup complete! Structure:")
    print(f"{DATASET_DIR}/day1, {DATASET_DIR}/day2, {DATASET_DIR}/day3")

if __name__ == "__main__":
    setup_dataset()
