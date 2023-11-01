"""Utils for PyTorch Geometric Data objects."""
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

def create_sequence_adjacent_edge_index(
    num_nodes: int,
    directed: bool = False,
    self_loops: bool = False,
    sequence_adjacency_range: int = 1,
) -> torch.Tensor:
    """ 
    Generate edge index for adjacent nodes in sequence.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    directed : bool, optional
        Whether the graph is directed or not, by default False
        # TODO 
    self_loops : bool, optional
        Whether to include self loops, by default False
        # TODO
    sequence_adjacency_range : int, optional
        The range of sequence adjacency, by default `1`. 
        For example, if `sequence_adjacency_range = 2`, then
        a given node i will be connected to nodes i-1, i+1, i-2, i+2.

    Returns
    -------
    torch.Tensor
        Edge index for adjacent nodes in sequence.

    """
    
    for i in range(sequence_adjacency_range):
        k = i + 1
        adjacent_edge_index = torch.stack([torch.arange(num_nodes - k), torch.arange(k, num_nodes)])
        if i == 0:
            edge_index = adjacent_edge_index
        else:
            edge_index = torch.cat([edge_index, adjacent_edge_index], dim=1)

    return edge_index

def replace_pt_edge_index_with_adjacent(
    g: torch_geometric.data.Data,
    sequence_adjacency_range: int = 2,
) -> torch_geometric.data.Data:
    """ 
    Replace edge index with adjacent edge index for a given PyTorch
    Geometric graph object. 

    This is used to generate controls for graphs to remove connections
    to nodes distant in sequence. 

    Parameters
    ----------
    g : torch_geometric.data.Data
        PyTorch Geometric graph object.
    sequence_adjacency_range : int, optional
        The range of sequence adjacency, by default `2`. 
        For example, if `sequence_adjacency_range = 2`, then
        a given node i will be connected to nodes i-1, i+1, i-2, i+2.
    
    Returns
    -------
    torch_geometric.data.Data
        PyTorch Geometric graph object with edge index replaced with
        adjacent edge index.
    
    """
    num_nodes = g.num_nodes
    edge_index = create_sequence_adjacent_edge_index(
        num_nodes=num_nodes,
        sequence_adjacency_range=sequence_adjacency_range,
    )
    g.edge_index = edge_index
    g = g.coalesce()
    return g

def reset_pt_edge_indexes(
    from_dir: Path,
    to_dir: Path,
    sequence_adjacency_range: int = 2, 
    uniprot_ids: List[str] = None,
    extension: str = ".pt",
    overwrite: bool = False, 
):
    """ 
    For a PyTorch geometric dataset, reset the `edge_index` attribute 
    of each graph to be sequential (sequence adjacent residues). 

    Parameters
    ----------
    from_dir : Path
        Path to directory containing the original dataset.
    to_dir : Path
        Path to directory to save the updated dataset.
    sequence_adjacency_range : int, optional
        The range of sequence adjacency, by default `2`. 
        For example, if `sequence_adjacency_range = 2`, then
        a given node i will be connected to nodes i-1, i+1, i-2, i+2.
    uniprot_ids : List[str], optional
        List of UniProt IDs to filter the dataset by, by default None.
    extension : str, optional
        File extension for the dataset processed files, by default ".pt".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.

    Returns
    -------
    None

    """
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
        
        # Replace `edge_index`
        data = replace_pt_edge_index_with_adjacent(
            g=data,
            sequence_adjacency_range=sequence_adjacency_range,
        )   
        torch.save(data, outfile)


def add_labels_to_dataset(
    from_dir: Path,
    to_dir: Path,
    uniprot_ids: List[str] = None,
    extension: str = ".pt",
    indexes_dict: Dict[str, Any] = None, 
    overwrite: bool = False, 
    skip_if_exists: bool = True, 
): 
    """ 
    Add labels to a dataset of PyTorch Geometric graph objects.

    Parameters
    ----------
    from_dir : Path
        Path to directory containing the original dataset.
    to_dir : Path
        Path to directory to save the updated dataset.
    uniprot_ids : List[str], optional
        List of UniProt IDs to filter the dataset by, by default None.
    extension : str, optional
        File extension for the dataset processed files, by default ".pt".
    indexes_dict : Dict[str, Any], optional
        Dictionary mapping UniProt IDs to indexes, by default None.
        Each entry should map a UniProt ID to a dictionary with the following
        keys: 
            - "idx": The index tensor of the nodes with labels. 
            - "y": The label tensor with specific labels for each node.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    skip_if_exists : bool, optional
        Whether to skip files that already contain `y_index` and `y` attributes,
        by default True.
    
    """
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