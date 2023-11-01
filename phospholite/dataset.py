
import glob

from pathlib import Path
from typing import Dict, List, Union, Any

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial

from phospholite.utils.io import load_index_dict
from phospholite.ml.dataset import PhosphositeGraphDataset

from phospholite import DATASET_DIR, INDEX_DICT_PATH
def get_dataset(
    root_dir: Path = DATASET_DIR / "protein_graph_dataset",
    index_dict: Any = None, 
    index_dict_path: Path = INDEX_DICT_PATH,
    verbose: bool = True, 
    uniprot_ids: List[str] = None,
    pre_labelled: bool = False, # Switch this to True if we want to avoid labelling the dataset on the fly. 
    first_n: int = None, 
): 

    if index_dict is None: 
        if index_dict_path is not None:
            index_dict = load_index_dict(filepath=index_dict_path)
        # If is None, assume our dataset already has labels. 

    long_interaction_threshold = 5 # seq positions 
    edge_threshold_distance = 6.0 # Å
    new_edge_funcs = {"edge_construction_functions": [
        partial(
        add_distance_threshold, long_interaction_threshold=long_interaction_threshold, threshold=edge_threshold_distance)
    ]}
    config = ProteinGraphConfig(
        granularity="CA",
        **new_edge_funcs,
    )
    from graphein.ml.conversion import GraphFormatConvertor

    columns = [
        "b_factor",
        "name",
        "edge_index",
        "x", # T5 per-residue embedding
    ]
    convertor = GraphFormatConvertor(
        src_format="nx", dst_format="pyg", verbose="gnn",
        columns=columns,
    )

    # List of functions that consume a nx.Graph and return a nx.Graph. Applied to graphs after construction but before conversion to pyg
    #from phosphosite.graphs.pyg import add_per_residue_embedding
    graph_transforms = [
        #add_per_residue_embedding,
    ]

    """
    Create dataset.
    """
    # Load in the actual dataset (i.e. processed filenames)
    if uniprot_ids is None:
        processed_filenames = [Path(a).stem for a in glob.glob(str(root_dir / "processed" / "*.pt"))]
        uniprot_ids_to_use = processed_filenames
        if verbose: print(f"Using {len(processed_filenames)} processed files.")
    else:
        uniprot_ids_to_use = uniprot_ids


    naked_path = DATASET_DIR / "naked_proteins.txt"
    if naked_path.exists():
        with open(naked_path, "r") as f:
            naked_proteins = f.read().splitlines()
            if verbose: print(f"Ignoring {len(naked_proteins)} naked proteins.")
        # Filter naked proteins (i.e. no sites on them)
        uniprot_ids_to_use = [u for u in uniprot_ids_to_use if u not in naked_proteins]


    if index_dict is not None:
        uniprot_ids_to_use = [u for u in uniprot_ids_to_use if u in index_dict.keys()] # avoid KeyError in index_dict
    if verbose: print(f"Using {len(uniprot_ids_to_use)} uniprot ids.")
    kwargs = dict(
        root=root_dir,
        graphein_config=config, 
        graph_transformation_funcs=graph_transforms,
        graph_format_convertor=convertor,
        pre_transform=None, # before saved to disk , after PyG conversion 
        pre_filter=None,    # whether it will be in final dataset
    )
    if first_n is not None:
        uniprot_ids_to_use = uniprot_ids_to_use[:first_n]
    ds = PhosphositeGraphDataset(
        uniprot_ids=uniprot_ids_to_use,
        y_label_map=index_dict if not pre_labelled else None,
        **kwargs,
    )
    return ds

