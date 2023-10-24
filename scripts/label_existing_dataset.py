"""Label an existing dataset with a dictionary of labels."""
import click as ck
import glob
import re
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch 
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT

import torch.nn as nn
import torch.nn.functional as F


from typing import Dict, List, Union
from pathlib import Path
from functools import reduce
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping, ModelCheckpoint

from phospholite import SAVED_MODEL_DIR, DATASET_DIR
from phospholite.model import PhosphoGAT
from phospholite.ml import get_dataloader_split
from phospholite.ml.dataset import PhosphositeGraphDataset 
from phospholite.utils.io import load_index_dict
from phospholite import INDEX_DICT_PATH



dataset_root_dir =  DATASET_DIR / "protein_graph_dataset"


"""Set random seed"""
torch.manual_seed(0) 
np.random.seed(0)





# TODO: 
# - run with 3 diff seeds to provide uncertainty estimates


@ck.command()
@ck.option(
    "--source",
    "-s",
    # TODO: 
    # for now use str representation of path. 
    default=str(dataset_root_dir),
    help="Root directory for data.",
)
@ck.option(
    "--dest",
    "-d",
    help="Directory to contain labelled dataset.",
)
def main(
    source: str,
    dest: str = None,
):
    """Label an existing dataset with a dictionary of labels."""
    if dest is None:
        dest = source
    add_labels_to_dataset(
        from_dir=Path(source),
        to_dir=Path(dest),
    )


import glob
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Dict, Any

import glob
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Dict, Any
import torch_geometric

def add_labels_to_dataset(
    from_dir: Path,
    to_dir: Path,
    uniprot_ids: List[str] = None,
    extension: str = ".pt",
    indexes_dict: Dict[str, Any] = None, 
    overwrite: bool = False, 
    skip_if_exists: bool = True, 
): 
    
    # Get all `.pt` files in `from_dir`
    files = glob.glob(str(from_dir / "processed" / f"*{extension}"))
    files = [Path(f) for f in files]
    # Filter files by `uniprot_ids`
    if uniprot_ids is not None:
        files = [f for f in files if f.stem in uniprot_ids]
    
    files = sorted(files)

    # Iterate over all files
    pbar = tqdm(files)
    for filename in pbar:

        outfile = to_dir / "processed" / f"{filename.stem}{extension}"
        if outfile.exists() and not overwrite:
            continue

        # Update progress bar description
        pbar.set_description(f"{filename.stem}")
        
        data = torch.load(filename)

        if not isinstance(data, torch_geometric.data.Data):
            continue

        assert data.name == filename.stem, f"{data.name} != {filename.stem}"
        if data.name not in indexes_dict.keys():
            continue
        uniprot_id = data.name

        # check if attributes already set. 
        if hasattr(data, "y") and hasattr(data, "y_index"):
            if skip_if_exists:
                continue
        
        data.y          = indexes_dict[uniprot_id]["y"]
        data.y_index    = indexes_dict[uniprot_id]["idx"]
            
        torch.save(data, outfile)