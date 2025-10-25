import os
import cv2
import pandas as pd
import numpy as np

def segment_plant_green(image):
    """
    Segments plant using green color detection in HSV space.
    Returns binary mask where plant pixels are white (255).
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green colors (plants)
    # Lower and upper bounds for green in HSV
    lower_green1 = np.array([35, 40, 40])   # Light green
    upper_green1 = np.array([85, 255, 255]) # Dark green
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def analyze_growth(dataset_dir="dataset", results_file="models/results.csv"):
    """
    Computes plant area and height for all images in day1/day2/day3 folders.
    Uses green color segmentation to detect plants more accurately.
    """
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    data = []

    for day in sorted(os.listdir(dataset_dir)):
        day_path = os.path.join(dataset_dir, day)
        if not os.path.isdir(day_path) or day.startswith("."):
            continue  # skip files or hidden folders

        print(f"Processing {day}...")
        day_areas = []
        day_heights = []

        for img_name in sorted(os.listdir(day_path)):
            if img_name.startswith("."):
                continue  # skip hidden files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # skip non-image files

            img_path = os.path.join(day_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read {img_path}, skipping.")
                continue

            # Segment plant using green color detection
            mask = segment_plant_green(img)

            # Calculate area (number of plant pixels)
            area = np.sum(mask > 0)

            # Calculate height (vertical span of plant)
            plant_rows = np.where(np.sum(mask, axis=1) > 0)[0]
            if len(plant_rows) > 0:
                height = plant_rows[-1] - plant_rows[0] + 1
            else:
                height = 0

            day_areas.append(area)
            day_heights.append(height)

            data.append({
                "day": day,
                "image": img_name,
                "area": area,
                "height": height
            })

        # Print daily statistics
        if day_areas:
            avg_area = np.mean(day_areas)
            avg_height = np.mean(day_heights)
            print(f"  {day}: Avg Area={avg_area:.0f}, Avg Height={avg_height:.0f} (from {len(day_areas)} images)")

    df = pd.DataFrame(data)
    df.to_csv(results_file, index=False)
    print(f"✅ Analysis complete. Results saved to {results_file}")
    return df
