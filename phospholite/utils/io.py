"""Reading / writing files."""
import torch
import json


from pathlib import Path
from typing import Dict, List, Union, Any


def save_index_dict(
    index_dict: Dict[str, Dict[str, torch.Tensor]],
    filepath: Path,
    overwrite: bool = False,
) -> None:
    """ 
    Save a dictionary of indexes to a json file. 

    NOTE: convert Tensors to lists before saving. 
    """

    # Create a copy of the index_dict with lists instead of tensors

    idx_dict = index_dict.copy()

    for key, value in idx_dict.items():
        idx_dict[key]["idx"] = value["idx"].tolist()
        idx_dict[key]["y"] = value["y"].tolist()


    if filepath.exists() and not overwrite:
        raise ValueError(f"File already exists at {filepath}")
    with open(filepath, "w") as f:
        json.dump(idx_dict, f)    

def load_index_dict(
    filepath: Path,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """ 
    Load a dictionary of indexes from a json file. 

    NOTE: convert lists to Tensors after loading. 
    """
    with open(filepath, "r") as f:
        index_dict = json.load(f)    

    for key, value in index_dict.items():
        index_dict[key]["idx"] = torch.tensor(value["idx"])
        index_dict[key]["y"] = torch.tensor(value["y"])

    return index_dict