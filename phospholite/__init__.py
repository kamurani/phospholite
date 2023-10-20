from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset"
PHOSPHOLITE_DIR = PROJECT_DIR / "phospholite"

if __name__ == "__main__":
    print(PROJECT_DIR)