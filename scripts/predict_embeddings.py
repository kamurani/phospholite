"""Run predict step of specific model checkpoint."""
import click as ck
import glob
import torch 
import numpy as np
import glob
import torch_geometric

from typing import Dict, List, Union, Any, Optional, Tuple, Callable
from pathlib import Path
from tqdm import tqdm

import torch

from pathlib import Path 
from phospholite.model import PhosphoGAT
from phospholite import INDEX_DICT_PATH
from phospholite.utils.io import load_index_dict

from phospholite import DATASET_DIR




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
    "--root-dir", 
    default=DATASET_DIR / "protein_graph_dataset",
    help="Root directory for dataset.",
)
def main(
    source: str,
    root_dir: str,
):
    # Extract embeddings from model (Gat2? )

    root_dir = Path(root_dir)

    checkpoint = Path(source)

    kwargs = dict(
        get_embeddings=True,
    )

    # Check if GPU 
    kwargs["map_location"] = None if torch.cuda.is_available() else torch.device('cpu')

    if checkpoint.is_dir(): checkpoint = checkpoint / "checkpoints" / "best.ckpt"

    model = PhosphoGAT.load_from_checkpoint(
        checkpoint,
        **kwargs,
    )

    from phospholite.dataset import get_dataset
    from phospholite import DATASET_DIR
     # default

    ds = get_dataset(
        root_dir=root_dir,
        #uniprot_ids=uniprot_ids,
        pre_labelled=True,
    )
    batch_size = 2
    from torch_geometric.loader import DataLoader
    dl = DataLoader(ds[0:4], batch_size=batch_size) # for testing

    import pytorch_lightning as pl
    trainer = pl.Trainer()

    output = trainer.predict(model, dl)


    emb_array = np.array([o[-1] for o in output])

    from phospholite.utils import generate_output_dataframe

    df = generate_output_dataframe(output, columns=["uniprot_id", "site", "label", "embedding"])

    model_dir = checkpoint.parent 
    name = checkpoint.stem
    filepath = model_dir / f"{name}_embedding_data.tsv"
    df.to_csv(filepath, sep="\t", index=False)

    filepath = model_dir / f"{name}_embedding_array.npy"
    np.save(filepath, emb_array)


if __name__ == "__main__":
    main()