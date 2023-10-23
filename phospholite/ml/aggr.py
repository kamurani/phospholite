import torch
from typing import Optional, Tuple, Union, List
from torch import Tensor

from torch_geometric.nn import Aggregation

class MaskedAggregation(Aggregation):

    def __init__(self, aggr_method: str = "mean") -> None:
        super().__init__()
        self.aggr_method = aggr_method

    def forward(
        self, x: Tensor, 
        index: Optional[Tensor] = None, 
        ptr: Optional[int] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        mask: Optional[Tensor] = None,
        mask_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Custom aggregation method that takes a mask and applies it to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor
        index : Optional[Tensor], optional
            Index tensor, by default None
        ptr : Optional[int], optional
            Pointer, by default None
        dim_size : Optional[int], optional
            Dimension size, by default None
        dim : int, optional
            Dimension, by default -2
        mask : Optional[Tensor], optional
            Mask tensor, by default None.  This should be a one-hot mask where
            non-zero values indicate the nodes to be used in the aggregation.
        mask_index : Optional[Tensor], optional
            Mask index tensor, by default None.  This should be a tensor of indexes
            that indicate the nodes to be used in the aggregation.

        Returns
        -------
        Tensor
            Aggregated tensor

        """
        # TODO: only index is supported currently (not ptr)


        # If mask is one-hot: 
        if mask is not None:
            mask_index = torch.nonzero(mask, as_tuple=True)[0] # identical to nonzero(...).reshape(-1)
                                                                # also identical to torch.where(mask)

        # If mask is indexes: 
        if mask_index is not None:
            # Create one-hot mask
            #mask = torch.zeros(dim_size, dtype=torch.bool)
            #mask[mask_index] = True
            pass
        
        # Get only the nodes by indexing into batch and x 
        index = index[mask_index]
        x = x[mask_index]


        return self.reduce(x, index, ptr, dim_size, dim, reduce=self.aggr_method)