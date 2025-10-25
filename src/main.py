from src.analysis import analyze_growth
from src.visualization import plot_growth

if __name__ == "__main__":
    dataset_path = "dataset/"
    df = analyze_growth(dataset_path)
    plot_growth("models/results.csv")
