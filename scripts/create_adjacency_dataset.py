"""Label an existing dataset with a dictionary of labels."""
import click as ck
import glob
import torch 
import numpy as np
import glob
import torch_geometric

from typing import Dict, List, Union, Any, Optional, Tuple, Callable
from pathlib import Path
from tqdm import tqdm

from phospholite import INDEX_DICT_PATH
from phospholite.utils.io import load_index_dict

from phospholite.utils.pt import reset_pt_edge_indexes






@ck.command()
@ck.option(
    "--source",
    "-s",
    # TODO: 
    # for now use str representation of path. 
    default=None,
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

    """Reset edge indexes for a dataset."""
    if dest is None:
        dest = source

    reset_pt_edge_indexes(
        from_dir=Path(source),
        to_dir=Path(dest), 
        sequence_adjacency_range=2, 
    )
   

if __name__ == "__main__":
    main()



