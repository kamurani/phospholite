"""Load everything we need to start training and see if it works."""

import glob 

from pathlib import Path
from utils.io import save_index_dict, load_index_dict

from phospholite import DATASET_DIR



filepath = DATASET_DIR / "indexes_dict.json"
index_dict = load_index_dict(filepath=filepath)

"""load model"""
from ml import PhosphositeGraphDataset
from model import PhosphoGAT

root_dir = DATASET_DIR / "protein_graph_dataset"

verbose = True
processed_filenames = [Path(a).stem for a in glob.glob(str(root_dir / "processed" / "*.pt"))]
if verbose: print(f"Using {len(processed_filenames)} processed files.")

ds = PhosphositeGraphDataset(
    root=root_dir,
    uniprot_ids=processed_filenames,
    y_label_map=index_dict,
)
print(ds)