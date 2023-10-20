"""Load everything we need to start training and see if it works."""

from pathlib import Path


from utils.io import save_index_dict, load_index_dict

from phospholite import DATASET_DIR

filepath = DATASET_DIR / "indexes_dict.json"
index_dict = load_index_dict(filepath=filepath)

print(index_dict)