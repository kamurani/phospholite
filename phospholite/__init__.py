from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT_DIR / "dataset"
PHOSPHOLITE_DIR = PROJECT_ROOT_DIR / "phospholite"
INDEX_DICT_PATH = DATASET_DIR / "indexes_dict.json"

"""ML training / saved models"""
SAVED_MODEL_DIR = PROJECT_ROOT_DIR / "train" / "saved_models"

if __name__ == "__main__":
    print(PROJECT_ROOT_DIR)