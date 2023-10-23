import torch


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(
        self,
        input, 
        target,
        mask,
    ):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class MaskedBinaryCrossEntropy(torch.nn.Module):
    def __init__(
        self, 
        subtype: str = "bce", # or "logits"
        loss_measure: str = "sum" # "avg"
    ):
        self.subtype = subtype
        self.loss_measure = loss_measure
        super(MaskedBinaryCrossEntropy, self).__init__()
        
    
    def forward(
        self,
        input, 
        target,
        mask,
    ):
        if self.subtype == "bce":
            
            loss = torch.nn.functional.binary_cross_entropy(
                input=torch.flatten(input),
                target=torch.flatten(target),
                weight=torch.flatten(mask),
                reduction="none",
            )

        elif self.subtype == "logits":
            # BCE with logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=torch.flatten(input),
                target=torch.flatten(target),
                weight=torch.flatten(mask),
                reduction="none",
            )

        if self.loss_measure == "sum":
            result = torch.sum(loss)
        elif self.loss_measure == "avg":
            result = torch.sum(loss) / int(torch.sum(mask))
        return result