from __future__ import annotations


import networkx as nx
import pandas as pd
# STFU
import graphein
graphein.verbose(enabled=False)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Any

from loguru import logger as log
import torch
from tqdm import tqdm

from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graphs_mp
from graphein.protein.utils import (
    download_alphafold_structure,
    download_pdb,
    download_pdb_multiprocessing,
)
from graphein.utils.dependencies import import_message

try:
    import torch
    from torch_geometric.data import Data, Dataset, InMemoryDataset
except ImportError:
    import_message(
        "graphein.ml.datasets.torch_geometric_dataset",
        "torch_geometric",
        conda_channel="pyg",
        pip_install=True,
    )

from torch import Tensor


# TODO: 
# raise issue in pytorch_geometric about error when creating dataset. (with download)
# Note: we had to resolve this issue manually by manually creating a `raw` directory 
# and moving files across that the `download` method was meant to create.

class PhosphositeGraphDataset(Dataset):
    def __init__(
            self, 
            root: str, 
            paths: List[str] | None = None, 
            pdb_codes: List[str] | None = None, 
            uniprot_ids: List[str] | None = None, 
            graph_labels: List[Tensor] | None = None, 
            node_labels: List[Tensor] | None = None, 
            chain_selections: List[str] | None = None, 
            graphein_config: ProteinGraphConfig = ProteinGraphConfig(), 
            graph_format_convertor: GraphFormatConvertor = GraphFormatConvertor(src_format="nx", dst_format="pyg"), 
            graph_transformation_funcs: List[Callable[..., Any]] | None = None, 
            pdb_transform: List[Callable[..., Any]] | None = None, 
            transform: Callable[..., Any] | None = None, 
            pre_transform: Callable[..., Any] | None = None, 
            pre_filter: Callable[..., Any] | None = None, 
            num_cores: int = 16, 
            af_version: int = 2,

            y_label_map: Dict[Dict[str, Tensor]] | None = None,
            
         
        ):
        if uniprot_ids is None:
            # Use everything provided in the label dictionary.
            uniprot_ids = list(y_label_map.keys())
        self.uniprot_ids = uniprot_ids

        self.examples: Dict[int, str] = dict(enumerate(self.uniprot_ids))

        if graph_labels is not None:
            self.graph_label_map = dict(enumerate(graph_labels))
        else:
            self.graph_label_map = None

        if node_labels is not None:
            self.node_label_map = dict(enumerate(node_labels))
        else:
            self.node_label_map = None

        if chain_selections is not None:
            self.chain_selection_map = dict(enumerate(chain_selections))
        else:
            self.chain_selection_map = None

        self.root = root
        self.y_label_map = y_label_map
        # Configs
        self.config = graphein_config
        self.graph_format_convertor = graph_format_convertor
        self.num_cores = num_cores
        self.pdb_transform = pdb_transform
        self.graph_transformation_funcs = graph_transformation_funcs
        self.af_version = af_version

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.config.pdb_dir = Path(self.raw_dir)
        
    
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files in the dataset."""
        # Return the uniprot_ids in raw_dir
        return [f"{uniprot_id}.pdb" for uniprot_id in self.uniprot_ids]
    

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files to look for"""
        return [f"{uniprot_id}.pt" for uniprot_id in self.uniprot_ids]
    
    def download(self):
        pass 

    def process(self):
        """Processes structures from files into PyTorch Geometric Data."""
        # Preprocess PDB files
        if self.pdb_transform:
            self.transform_pdbs()

        idx = 0
        # Chunk dataset for parallel processing
        chunk_size = 128

        def divide_chunks(l: List[str], n: int = 2) -> Generator:
            for i in range(0, len(l), n):
                yield l[i : i + n]

        # chunks = list(divide_chunks(self.structures, chunk_size))
        chunks: List[int] = list(
            divide_chunks(list(self.examples.keys()), chunk_size)
        )

        for chunk in tqdm(chunks):
            pdbs = [self.examples[idx] for idx in chunk]
            # Get chain selections
            if self.chain_selection_map is not None:
                chain_selections = [
                    self.chain_selection_map[idx] for idx in chunk
                ]
            else:
                chain_selections = ["all"] * len(chunk)

            # Create graph objects
            file_names = [f"{self.raw_dir}/{pdb}.pdb" for pdb in pdbs]

            graphs = construct_graphs_mp(
                path_it=file_names,
                config=self.config,
                chain_selections=chain_selections,
                return_dict=False,
            )
            old_len = len(graphs)
            graphs = [g for g in graphs if g is not None]
            if old_len - len(graphs) > 0:
                print("number of None graphs: ", old_len - len(graphs))

            if self.graph_transformation_funcs is not None:
                graphs = [self.transform_graphein_graphs(g) for g in graphs]

            # Convert to PyTorch Geometric Data
            converted = []
            for i, g in enumerate(graphs):
                try:
                    converted_graph = self.graph_format_convertor(g)
                    converted.append(converted_graph)
                except:
                    
                    tqdm.write(f"failed to convert graph {g}")
                    continue
                
                
                
            #graphs = [self.graph_format_convertor(g) for g in graphs]
            graphs = converted

            # Assign labels
            if self.graph_label_map:
                labels = [self.graph_label_map[idx] for idx in chunk]
                for i, _ in enumerate(chunk):
                    graphs[i].graph_y = labels[i]
            if self.node_label_map:
                labels = [self.node_label_map[idx] for idx in chunk]
                for i, _ in enumerate(chunk):
                    graphs[i].graph_y = labels[i]

            data_list = graphs

            del graphs

            if self.pre_filter is not None:
                data_list = [g for g in data_list if self.pre_filter(g)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # TODO: handle chain selections? 

            # for now just save each graph by its `name`
            for i, g in enumerate(data_list):
                
                torch.save(
                    g,
                    os.path.join(self.processed_dir, f"{g.name}.pt"),
                )
                
        
        # Update valid modiform_ids 
        # get all .pt uniprot_ids 
        self.uniprot_ids = [
            u
            for u in self.uniprot_ids
            if os.path.exists(os.path.join(self.processed_dir, f"{u}.pt"))
        ]
        
        # Don't have to update the y_label_map because it's a dict
        # and the dataset can only be indexed into the valid uniprot_ids 
        # (i.e. valid uniprot_id means graph has been constructed)
        # and we already pre-filter for making sure the unipro_id sequence
        # is valid and maps with our sites.

    
    def len(self) -> int:
        """
        Returns length of graphs.
        """
        return len(self.uniprot_ids)

    def get(self, idx: int):
        """
        Returns PyTorch Geometric Data object for a given index.

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        if isinstance(idx, str):
            # assume uniprot_id 
            uniprot_id = idx
        else:
            uniprot_id = self.uniprot_ids[idx]
        data = torch.load(
            os.path.join(self.processed_dir, f"{uniprot_id}.pt")
        )
        assert isinstance(data, Data), f"{uniprot_id}: Data is not a PyTorch Geometric Data object. Got {type(data)}."
        assert uniprot_id == data.name, f"Uniprot ID '{uniprot_id}' does not match data.name '{data.name}' at index {idx}."
        
        if self.y_label_map is not None:
            data.y_index    = self.y_label_map[uniprot_id]["idx"]
            data.y          = self.y_label_map[uniprot_id]["y"]

        #data.phosphosite_index = torch.tensor(site_indexes, dtype=torch.long)
        # NOTE: not necessary to store any extra information.
        return data

    def transform_graphein_graphs(self, graph: nx.Graph):
        for func in self.graph_transformation_funcs:
            graph = func(graph)
        return graph