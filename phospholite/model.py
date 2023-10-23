"""ML models for predicting phosphorylation sites."""

# pytorch geometric and pytorch lightning
import numpy as np 
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, Sequential as PyGSequential
from typing import Callable, List, Tuple, Union, Optional

from torchmetrics import Accuracy
accuracy = Accuracy(task="binary")

from phospholite.ml import calculate_masked_accuracy, calculate_masked_f1


# import metrics from torch
#from torchmetrics import Accuracy, Precision, Recall, F1, AUROC, ConfusionMatrix


SEQUENCE_EMBEDDING_SIZE = 1024
from phospholite.ml import MaskedBinaryCrossEntropy, MaskedMSELoss



class PhosphoGAT(pl.LightningModule):
    """Graph Attention Network for predicting phosphorylation sites.
    
    Parameters
    ----------

    """
    def __init__(
        self, 

        # Model params
        node_embedding_size: int = SEQUENCE_EMBEDDING_SIZE, 
        num_heads: int = 8, 
        dropout: float = 0.1,

        # Training params 
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_workers: int = 8,
        num_epochs: int = 200,


        train_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        test_dataset: Dataset = None,

        loss_func: Callable = MaskedBinaryCrossEntropy(),
        accuracy: Callable = calculate_masked_accuracy,

        prog_bar: bool = True,

    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.out_heads = 1 

        self.loss_func = loss_func
        self.accuracy = accuracy

        self.prog_bar = prog_bar

        """Model parameters."""
        hidden_size = node_embedding_size // num_heads  

        out_size = hidden_size*num_heads # change this to be less perhaps?

        # TODO: 
        # - test versions of allowing more or less dimensions (128 up to 1024 (same as original embeddings)) 
        # in the output layers of the Conv

        """Layers"""
        self.conv1 = GATv2Conv(node_embedding_size, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_size*num_heads, out_size, heads=self.out_heads, dropout=dropout)

        self.convolutions = PyGSequential(
            "x, edge_index", [
                # Dropout before the conv layers? Hmm.
                (nn.Dropout(dropout), "x -> x"),
                (self.conv1, "x, edge_index -> x"),
                nn.SELU(),
                nn.Dropout(dropout),
                (self.conv2, "x, edge_index -> x"),
                nn.SELU(),
            ]
        )
        # After the conv layers, we should ideally have a learnt representation of the structural / sequence 
        # information of the protein per each site. We can then apply the fully connected layer to each
        # node to get a phosphosite prediction. 

        # TODO: 
        # use b_factor as a kind of mask? 

        """Classification"""

        # Several fully connected layers with dropout and selu 
        self.classifier = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            # binary classification i.e. sigmoid
            nn.Sigmoid(),
        )

    def forward(
        self, 
        protein_data, 

    ) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        protein_data : torch_geometric.data.Data
            The input data object.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, output_dim)

        """
        
        x, edge_index, batch = protein_data.x, protein_data.edge_index, protein_data.batch


        x = self.convolutions(x, edge_index)
        #pooled = global_add_pool(x, batch)
       
        # Apply fully connected layers to each node
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step for the model.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss tensor.

        """
        x = batch 
        y_sparse = x.y 
        y_index = x.y_index
        y_index = y_index.to(torch.long)
        y_hat = self(x)

        # Flatten 
        y_hat = torch.flatten(y_hat)
        y_sparse = torch.flatten(y_sparse)
        y_index = torch.flatten(y_index)

        # Use `y_index` to create mask 
        mask = torch.zeros_like(y_hat)
        mask[y_index] = 1
        y = torch.zeros_like(y_hat)
        y[y_index] = y_sparse


        loss = self.loss_func(y_hat, y, mask)
        acc = self.accuracy(y_hat, y, mask)
        self.log("train_loss", loss, prog_bar=self.prog_bar)
        self.log("train_acc", acc, prog_bar=self.prog_bar)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch 
        y_sparse = x.y 
        y_index = x.y_index
        y_index = y_index.to(torch.long)
        y_hat = self(x)

        # Flatten 
        y_hat = torch.flatten(y_hat)
        y_sparse = torch.flatten(y_sparse)
        y_index = torch.flatten(y_index)
        # Use `y_index` to create mask 
        mask = torch.zeros_like(y_hat)
        mask[y_index] = 1
        y = torch.zeros_like(y_hat)
        y[y_index] = y_sparse

        loss = self.loss_func(y_hat, y, mask)
        acc = self.accuracy(y_hat, y, mask)
        self.log("val_loss", loss, prog_bar=self.prog_bar)
        self.log("val_acc", acc, prog_bar=self.prog_bar)

    def test_step(self, batch, batch_idx):
        x = batch 
        y_sparse = x.y 
        y_index = x.y_index
        y_index = y_index.to(torch.long)
        y_hat = self(x)

        # Flatten 
        y_hat = torch.flatten(y_hat)
        y_sparse = torch.flatten(y_sparse)
        y_index = torch.flatten(y_index)
        # Use `y_index` to create mask 
        mask = torch.zeros_like(y_hat)
        mask[y_index] = 1
        y = torch.zeros_like(y_hat)
        y[y_index] = y_sparse

        loss = self.loss_func(y_hat, y, mask)
        acc = self.accuracy(y_hat, y, mask)

        # F1 score
        f1 = calculate_masked_f1(y_hat, y, mask, average="weighted")

        self.log("test_loss", loss, prog_bar=self.prog_bar)
        self.log("test_acc", acc, prog_bar=self.prog_bar)
        self.log("test_f1", f1, prog_bar=self.prog_bar)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch 
        y_sparse = x.y 
        y_index = x.y_index
        y_index = y_index.to(torch.long)
        y_hat = self(x)

        # Flatten 
        y_hat = torch.flatten(y_hat)
        y_sparse = torch.flatten(y_sparse)
        y_index = torch.flatten(y_index)

        # Use `y_index` to create mask 
        mask = torch.zeros_like(y_hat)
        mask[y_index] = 1
        y = torch.zeros_like(y_hat)
        y[y_index] = y_sparse

        """Index with batch idx and selected y values
        """
        
        batch_idx = batch.batch 
        batch_idx = batch_idx[y_index]          # Select the batch_idx for each y value (indexing with `y_index`)

        
        if isinstance(batch.name, torch.Tensor):
            # detach 
            ids = batch.name.detach().cpu().numpy()
            uniprot_ids = ids[batch_idx].tolist()
        elif isinstance(batch.name, list):
            ids = np.array(batch.name)
            uniprot_ids = ids[batch_idx].tolist()
        else: 
            raise ValueError(f"batch.name is of type {type(batch.name)}") 

        node_id_flat = np.array([item for sublist in batch.node_id for item in sublist], dtype=str)
        node_ids = node_id_flat[y_index].tolist()

        y_values = y_sparse.tolist()

        y_hat = y_hat[y_index]
        y_hat_values = y_hat.tolist()
        prediction_values = np.round(y_hat.detach().cpu().numpy()).tolist() # thresholded
        return list(zip(uniprot_ids, node_ids, y_values, prediction_values, y_hat_values,))


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5, verbose=True, mode="min")
        return dict(
            optimizer=optim,
            lr_scheduler=scheduler,
            monitor="train_loss",
        )
