
#%%
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import monai
import pdb



#%%


class WeightedBCELoss(nn.Module):

    def __init__(self, weight = 100, sigmoid=True) -> None:
        super().__init__()
        self.weight = weight
        self.sigmoid = sigmoid

    def __call__(self, pred: Tensor, target: Tensor):
        if self.sigmoid:
            loss = nn.BCEWithLogitsLoss(weight=torch.where(target!=0, self.weight, 1.0))
        else:
            loss = nn.BCELoss(weight=torch.where(target!=0, self.weight, 1.0))
        return loss(pred, target)

class FocalDiceLoss(nn.Module):

    def __init__(self, gamma = 2) -> None:
        super().__init__()
        #print(super(FocalDiceLoss)._modules)
        self.gamma = gamma
        self.eps: float = 1e-6

    def __call__(self, pred: Tensor, target: Tensor):
        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(pred)))
        if not len(pred.shape) == 5:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}".format(input.shape))
        if not pred.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}".format(pred.shape, pred.shape))
        if not pred.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(pred.device, target.device))

        pred_sig = torch.sigmoid(pred)
        p_t = pred * target + (1 - pred) * (1 - target)
        w_local = ((1 - p_t) ** self.gamma)
        dims = (1, 2, 3, 4)
        intersection = torch.sum(w_local*(pred_sig * target), dims)
        cardinality = torch.sum(w_local*(pred_sig + target), dims)

        dice_score = 2. * intersection / (cardinality + self.eps)


        return torch.mean(1. - dice_score)



class DiceCrossEntropyLoss(nn.Module):

    def __init__(self, ratioce = 0.5, cepositiveweight=0.99) -> None:
        super().__init__()
        self.ratioce = ratioce
        self.cepositiveweight = cepositiveweight
        assert self.cepositiveweight < 1 and cepositiveweight >= 0
        self.dice = monai.losses.Dice()
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1/(1-self.cepositiveweight), device='cuda' if torch.cuda.is_available() else 'cpu'))

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        target = torch.softmax(target, dim=1)
        return self.ce(pred,target)/self.cepositiveweight + self.dice(pred,target)

class MixLoss(nn.Module):

    def __init__(self, losses:list, weights:list) -> None:
        super().__init__()
        self.weights = weights
        self.losses = losses
        assert len(self.weights) == len(self.losses)

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        target = torch.softmax(target, dim=1)
        final_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            final_loss += weight * loss(pred, target)
        return final_loss

class MixSweepLoss(nn.Module):

    def __init__(self, losses:list, weights:list) -> None:
        super().__init__()
        self.weights = weights
        self.losses = losses
        assert len(self.weights) == len(self.losses)

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        target = torch.softmax(target, dim=1)
        final_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            final_loss += weight * loss(pred, target)
        return final_loss

    def update_weights(self, weights:list):
        assert len(self.weights) == len(self.losses)
        self.weights = weights

    def update_ratio(self, ratio_of_second):
        assert len(self.losses) == 2
        assert ratio_of_second >= 0 and ratio_of_second <= 1
        new_weights = [1-ratio_of_second, ratio_of_second]
        self.update_weights(new_weights)

    def get_weights(self):
        return self.weights

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-05, sigmoid=True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def __call__(self, y_true, y_pred) -> Tensor:

        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1. - ((2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth))

class FocalLoss(nn.Module):
    def __init__(self, 
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none") -> None:

        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self,inputs: torch.Tensor, targets: torch.Tensor):
    
        p = torch.sigmoid(inputs)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss



# %%
