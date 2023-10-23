from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import List, Dict, Union, Tuple, Optional

import torch



def get_dataloader_split(
    ds: torch.utils.data.Dataset,   
    train_test_ratio: float = 0.8,
    train_valid_ratio: float = 0.8,
    train_batch_size: int = 32,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ 
    Splits a dataset into train, valid, test dataloaders.

    Parameters
    ----------
    ds : torch.utils.data.Dataset
        Dataset to split into train, valid, test dataloaders.
    train_test_ratio : float, optional
        Ratio of train+valid to test data, by default 0.8
    train_valid_ratio : float, optional
        Ratio of train to valid data, by default 0.8
    train_batch_size : int, optional
        Batch size for training, by default 32
    batch_size : int, optional
        Batch size for validation and testing, by default 32
    num_workers : int, optional
        Number of workers for dataloading, by default 8
    
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        Train, valid, test dataloaders.
    
    """



    train_size = int(train_test_ratio * len(ds))
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, len(ds) - train_size])

    train_size = int(train_valid_ratio * len(train_dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, 
        num_workers=num_workers, 
        shuffle = True, drop_last = True, )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers) # 32
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,)   # 32
    return train_loader, valid_loader, test_loader