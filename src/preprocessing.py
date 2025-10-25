import cv2
import numpy as np
import os

def load_images_from_folder(folder_path):
    """
    Loads all valid images from a folder in sorted order.
    Skips folders, hidden files, and non-image files.
    Returns a list of images (numpy arrays).
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path) or filename.startswith("."):
            continue  # skip folders and hidden files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # skip non-image files

        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"[WARN] Could not read {file_path}, skipping.")
    return images


def resize_image(image, size=(512, 512), keep_aspect_ratio=True):
    """
    Resizes an image to the specified size.
    If keep_aspect_ratio=True, pads the image to maintain aspect ratio.
    """
    if not keep_aspect_ratio:
        return cv2.resize(image, size)

    h, w = image.shape[:2]
    target_w, target_h = size

    # Compute scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create black canvas and paste resized image centered
    canvas = np.zeros((target_h, target_w, 3), dtype='uint8') \
             if len(image.shape) == 3 else np.zeros((target_h, target_w), dtype='uint8')

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return canvas


def resize_folder(folder_path, size=(512, 512), keep_aspect_ratio=True):
    """
    Resizes all valid images in a folder to the specified size and overwrites them.
    Skips hidden files, folders, and non-image files.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path) or filename.startswith("."):
            continue  # skip folders and hidden files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # skip non-image files

        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARN] Could not read {file_path}, skipping.")
            continue

        img_resized = resize_image(img, size=size, keep_aspect_ratio=keep_aspect_ratio)
        cv2.imwrite(file_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    print(f"✅ All valid images in '{folder_path}' resized to {size}")


def resize_dataset(dataset_dir="dataset", size=(512, 512), keep_aspect_ratio=True):
    """
    Resizes all images in all subfolders of dataset_dir.
    """
    for subfolder in sorted(os.listdir(dataset_dir)):
        subfolder_path = os.path.join(dataset_dir, subfolder)
        if os.path.isdir(subfolder_path) and not subfolder.startswith("."):
            resize_folder(subfolder_path, size=size, keep_aspect_ratio=keep_aspect_ratio)
