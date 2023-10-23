import torch

from sklearn.metrics import f1_score 
from torchmetrics import Accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy = Accuracy(task="binary").to(device)

def calculate_masked_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Calculate the accuracy of the model's predictions.

    Only considers the predictions for the masked residues, 
    i.e. where the mask is 1.0.

    Parameters
    ----------
    input : torch.Tensor
        The model's predictions.
    target : torch.Tensor
        The target values.
    mask : torch.Tensor
        The mask tensor.

    Returns
    -------
    torch.Tensor
        The accuracy.

    """
    # Get indexes where mask is 1.0

    # Detach and copy to cpu 
    input = input.detach().cpu()
    target = target.detach().cpu()
    if mask is not None:
        mask = mask.detach().cpu() 
        mask_idx = torch.where(mask == 1)
        input = input[mask_idx]
        target = target[mask_idx]
    
    acc = accuracy(input, target)
    return acc
    
def calculate_masked_f1(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    **kwargs,
) -> float:
    """
    Calculate the F1 score of the model's predictions.

    Only considers the predictions for the masked residues, 
    i.e. where the mask is nonzero. 

    Parameters
    ----------
    input : torch.Tensor
        The model's predictions.
    target : torch.Tensor
        The target values.
    mask : torch.Tensor
        The mask tensor.
    **kwargs
        Keyword arguments to pass to `sklearn.metrics.f1_score`.

    Returns
    -------
    float
        The F1 score.

    """
    # Detach 
    input = input.detach().cpu()
    target = target.detach().cpu()

    y_pred = (input > 0.5).int()

    def get_masked(tensor):
        tensor = tensor[indexes]
        return tensor.detach().cpu().numpy()
    
    if mask is not None:
        mask = mask.detach().cpu()
        indexes = torch.nonzero(mask, as_tuple=True)[0] 
        target        = get_masked(target)
        y_pred   = get_masked(y_pred) 

    return f1_score(target, y_pred, **kwargs)



