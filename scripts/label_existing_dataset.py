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
@ck.option(
    "--index",
    "-i",
    "index_path",
    default=INDEX_DICT_PATH,
    help="Path to index dictionary.",
)
def main(
    source: str,
    dest: str = None,
    index_path: Path = None, 
):
    index_path = Path(index_path)
    """Label an existing dataset with a dictionary of labels."""
    if dest is None:
        dest = source
    add_labels_to_dataset(
        from_dir=Path(source),
        to_dir=Path(dest),
        indexes_dict=load_index_dict(index_path),
    )

if __name__ == "__main__":
    main()



