from src.preprocessing import resize_folder
from src.analysis import analyze_growth
from src.visualization import plot_growth
import os
import sys
import subprocess

# Add src folder to path
sys.path.append(os.path.join(os.getcwd(), "src"))


def main():
    print("🌱 PLANT GROWTH MONITORING SYSTEM 🌱\n")

    # Step 1: Check if dataset exists
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir) or not any(os.scandir(dataset_dir)):
        print("[INFO] Dataset not found or empty. Running setup_dataset.py ...")
        subprocess.run(["python", "setup_dataset.py"], check=True)
    else:
        print("[INFO] Dataset already present.")

    # Step 2: Resize all images in dataset (aspect ratio preserved)
    print("\n🔄 Resizing all images in dataset to 512×512 (aspect ratio preserved) ...")
    for day in os.listdir(dataset_dir):
        day_path = os.path.join(dataset_dir, day)
        if os.path.isdir(day_path):
            resize_folder(day_path, size=(512, 512), keep_aspect_ratio=True)

    # Step 3: Analyze growth (compute area + height)
    print("\n📊 Analyzing plant growth ...")
    results_file = "models/results.csv"
    df = analyze_growth(dataset_dir, results_file=results_file)

    # Step 4: Visualize growth trends
    print("\n📈 Visualizing growth trends ...")
    plot_growth(results_file)

    print("\n✅ Project completed successfully!")
    print(f"Results saved in '{results_file}' and graphs displayed.")


if __name__ == "__main__":
    main()
