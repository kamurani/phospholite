
"""Training script for phosphosite predictor."""

# TODO:
# - have 'setup' function that downloads the Pretrained embeddings from uniprot and puts it in correct directory etc. 
# - so we don't have to manually upload directory every time to cloud computing

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
from phospholite.utils import generate_output_dataframe
from phospholite import INDEX_DICT_PATH



dataset_root_dir =  DATASET_DIR / "protein_graph_dataset"


"""Set random seed"""
torch.manual_seed(0) 
np.random.seed(0)





# TODO: 
# - run with 3 diff seeds to provide uncertainty estimates


@ck.command()
@ck.option(
    "--root-dir",
    "-r",
    # TODO: 
    # for now use str representation of path. 
    default=str(dataset_root_dir),
    help="Root directory for data.",
)
@ck.option(
    "--model-dir",
    "-m",
    default=SAVED_MODEL_DIR,
    help="Directory to save model.",
)
@ck.option(
    "--dev/--no-dev",
    default=False,
    help="Run a single training batch for debugging.",
)
@ck.option(
    "--model-name",
    "-n",
    default="M1",
    help="Name of the model to train.",
)
@ck.option(
    "--epochs",
    "-e",
    default=200,
    help="Number of epochs to train.",
)
@ck.option(
    "--dryrun/--no-dryrun",
    default=False,
    help="Run a dryrun of the training script.",
)
@ck.option(
    "--verbose/--no-verbose",
    default=True,
    help="Print more output.",
)
@ck.option(
    "--num-workers",
    "-w",
    default=0,
    help="Number of workers to use for dataloader.",
)
@ck.option(
    "--batch-size",
    "-b",
    default=32,
    help="Batch size.",
)
@ck.option(
    "--first-n",
    default=None,
    help="First n proteins to train on.",
)
def main(
    root_dir: str = "",
    model_dir: str = "",
    dev: bool = False,
    model_name: str = "phosphogat",
    epochs: int = 200, 
    dryrun: bool = False,
    verbose: bool = False,
    num_workers: int = 0,
    batch_size: int = 32,
    first_n: int = None, 
):
    root_dir = Path(root_dir)
    model_dir = Path(model_dir)
    if verbose: 
        print(f"[Dataset] Using root directory: {root_dir}")
        print(f"[Model] Using root directory: {model_dir}")
    dropout = 0.1 
    batch_size = 64 # 100
    num_heads = 8 # 4 
    learning_rate = 0.001

    hidden_embedding_size = 256 

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}.")

    learning_rate = 0.001
    np.random.seed(42)
    #idx_all = np.arange(len(pyg_list))
    #np.random.shuffle(idx_all)

    """SETUP 
    - load graph construction functions etc.
    
    """
    from phospholite.dataset import get_dataset
    ds = get_dataset(
        root_dir=root_dir,
        index_dict_path=INDEX_DICT_PATH,
        verbose=verbose,
        first_n=first_n,
    )
    if first_n is not None:
        if verbose: print(f"Using first {first_n} proteins.")
    if verbose: print(ds)

    train_loader, valid_loader, test_loader = get_dataloader_split(
        ds, batch_size=batch_size, train_batch_size=batch_size,
        #num_workers=num_workers,
    )
    if verbose: print(train_loader, valid_loader, test_loader)

 
    model = PhosphoGAT(
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    """Train model."""
    # Early stopping 
    patience = 10
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=patience,   
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    

    trainer = Trainer(
        default_root_dir=model_dir,
        max_epochs=epochs,
        accelerator="auto", devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        fast_dev_run=dev,
    )
    trainer.fit(model, train_loader, valid_loader)

    all_dataloaders = [
        train_loader,
        valid_loader,
        test_loader,
    ]
  
    
    # evaluate on the model with the best validation set
    if dev: 
        if verbose: print(f"Dev mode: testing on test set for first model.")
        trainer.test(
            model=model,
            ckpt_path=None, 
            dataloaders=test_loader)
        data1 = trainer.predict(
            model, dataloaders=all_dataloaders)
    else:
        trainer.test(ckpt_path="best", dataloaders=test_loader)
        data1 = trainer.predict(
            dataloaders=all_dataloaders,
            ckpt_path="best",
        )
    
    
    # Data1 should be a `list` of `dict` objects for each dataloader. 
    # For now, assume we've passed just ONE dataloader. 

    """Save predictions."""
    for i, name in enumerate(["train", "valid", "test"]):
        #print(f"data1[{i}]: {data1[i]}")
        #output_df = generate_output_dataframe(data1[i][0])

        # ADDED ##########
        output = data1[i]
        from phospholite.utils import flatten_predictions
        output = flatten_predictions(output)

        df = generate_output_dataframe(output)
        #model_dir = checkpoint.parent
        filepath = model_dir / f"{name}_predictions_all.tsv"
        if verbose: print(f"Saving to {filepath} ...")
        df.to_csv(filepath, sep="\t", index=False)

        ##################


        #filepath = model_dir / f"phosphosite_predictions_{name}.csv"
        
        #output_df.to_csv(filepath, index=False, sep="\t")
    
    """Save embeddings."""
    ###########################################################
    batch_size = 256
    from torch_geometric.loader import DataLoader
    dl = DataLoader(ds, batch_size=batch_size) # for testing

    # Load model again 
    # checkpoint = model_dir / "best.ckpt"
    kwargs = dict(
        get_embeddings=True,
    )
    # Check if GPU 
    kwargs["map_location"] = None if torch.cuda.is_available() else torch.device('cpu')
    
    import pytorch_lightning as pl
    #trainer = pl.Trainer()

    # Make sure the `predict_step` will return embeddings by setting the param in the model internally.
    model.get_embeddings = True

    output = trainer.predict(
        model=model,
        dataloaders=dl,
        ckpt_path="best" if not dev else None,
    )
    from phospholite.utils import flatten_predictions
    output = flatten_predictions(output)

    generate_embeddings = True 
    if generate_embeddings:

        emb_array = np.array([o[-1] for o in output])
        df = generate_output_dataframe(output, columns=["uniprot_id", "site", "label", "embedding"])
        model_dir = checkpoint.parent 
        name = checkpoint.stem
        filepath = model_dir / f"{name}_embedding_data.tsv"
        df.to_csv(filepath, sep="\t", index=False)

        filepath = model_dir / f"{name}_embedding_array.npy"
        np.save(filepath, emb_array)

    print("Done.")
    # TODO

        



if __name__ == "__main__":
    main()