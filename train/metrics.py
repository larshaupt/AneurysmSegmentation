#%%
import train.utils as ut

from typing import Callable
import torch
from torch import Tensor
from sklearn.metrics import f1_score
from torchmetrics import Dice, Precision, Recall, AUROC
import numpy as np
import monai
import pdb

#%%
def convert_to_tensor(data):
    if not isinstance(data, Tensor):
        return Tensor(data)
    else:
        return data


class MultiClassDiceMetric():

    def __init__(self, include_background=True) -> None:
        self.include_background = include_background
        self.metric = monai.metrics.GeneralizedDiceScore(include_background = self.include_background, reduction="mean_batch")

    def __call__(self, pred: Tensor, target: Tensor):

        assert pred.ndim == 4 and target.ndim==4
        pred, target = (convert_to_tensor(pred)).to('cpu'), (convert_to_tensor(target)).to('cpu')
        pred, target = torch.softmax(pred, dim=0), torch.softmax(target, dim=0)
        pred = torch.nn.functional.one_hot(torch.argmax(pred, dim=0), pred.shape[0]).movedim(3,0) # binarizes the prediction tensor
        return self.metric(pred.view(1, *pred.shape), target.view(1, *target.shape)).item() # need to add extra dimension for function to accept it

class F2Score():
    def __init__(self, beta=0.5) -> None:
        self.beta=beta

    def __call__(self, pred: Tensor, target: Tensor):
        pred, target = convert_to_tensor(pred), convert_to_tensor(target)
        pred, target = ut.binarize(pred), ut.binarize(target)
        TP = (pred*target).sum()
        FN = (torch.where(pred==0, 1,0)*target).sum()
        FP = (torch.where(target==0, 1,0)*pred).sum()
        if 2*TP + FP + FN == 0:
            if TP == 0:
                return 1.0
            return 0.0
        return 2*TP / (2*TP + FP + FN)



class DiceMetric():

    def __init__(self) -> None:
        pass

    def __call__(self, pred: Tensor, target: Tensor):
        
        pred, target = convert_to_tensor(pred), convert_to_tensor(target)
        assert pred.shape[0] == 1 and target.shape[0] == 1 
        pred, target = ut.binarize(pred), ut.binarize(target)
        
        TP = (pred*target).sum()
        FN = (torch.where(pred==0, 1,0)*target).sum()
        FP = (torch.where(target==0, 1,0)*pred).sum()
        if 2*TP + FP + FN == 0:
            if TP == 0:
                return 1.0
            return 0.0
        return 2*TP / (2*TP + FP + FN)



class RecallMetric():
    def __init__(self) -> None:
        pass
    def __call__(self, pred: Tensor, target: Tensor):
        pred, target = convert_to_tensor(pred), convert_to_tensor(target)
        pred, target = ut.binarize(pred), ut.binarize(target)
        TP = (pred*target).sum()
        FN = (torch.where(pred==0, 1,0)*target).sum()

        if TP+FN == 0:
            if TP == 0:
                return 1.0
            return 0.0
            
        return TP / (TP+FN)


class PrecisionMetric():
    def __init__(self) -> None:
        pass
        
    def __call__(self, pred: Tensor, target: Tensor):
        pred, target = convert_to_tensor(pred), convert_to_tensor(target)
        pred, target = ut.binarize(pred), ut.binarize(target)
        TP = (pred*target).sum()
        FP = (torch.where(target==0, 1,0)*pred).sum()

        if TP+FP == 0:
            if TP == 0:
                return 1.0
            return 0.0
            
        return TP / (TP+FP)

def compute_TN(pred: Tensor, target: Tensor):
    pred, target = convert_to_tensor(pred), convert_to_tensor(target)
    pred, target = ut.binarize(pred).bool(), ut.binarize(target).bool()
    TN  = (~target*~pred).sum()
    return TN

def compute_FN(pred: Tensor, target: Tensor):
    pred, target = convert_to_tensor(pred), convert_to_tensor(target)
    pred, target = ut.binarize(pred).bool(), ut.binarize(target).bool()
    FN  = (target*~pred).sum()
    return FN

def compute_TP(pred: Tensor, target: Tensor):
    pred, target = convert_to_tensor(pred), convert_to_tensor(target)
    pred, target = ut.binarize(pred).bool(), ut.binarize(target).bool()
    TP  = (target*pred).sum()
    return TP

def compute_FP(pred: Tensor, target: Tensor):
    pred, target = convert_to_tensor(pred), convert_to_tensor(target)
    pred, target = ut.binarize(pred).bool(), ut.binarize(target).bool()
    FP  = (~target*pred).sum()
    return FP


class TargetLabelMetric():
    def __init__(self, metric, target_class) -> None:
        
        self.metric = metric
        self.target_class = target_class
        

    def __call__(self, pred: Tensor, target: Tensor):
        return self.metric(pred[self.target_class,:,:,:].unsqueeze(0), target[self.target_class,:,:,:].unsqueeze(0))


class MetricesStruct():
    def __init__(self, metrices, prefix:str = "") -> None:

        if isinstance(metrices, dict):
            self.metrices = metrices
        elif isinstance(metrices, Callable):
            self.metrices= {"metric": metrices}
        
        else:
            raise RuntimeError(f"Metrices has the wrong type {type(metrices)}")
        self.scores = dict.fromkeys(self.metrices, 0)
        for key in self.scores.keys():
            self.scores[key] = []
        self.count = 0
        self.average_scores = {}
        self.prefix = prefix

    def update(self, pred, target, el_name=""):

        for el_id in range(pred.shape[0]): 
            for name, func in self.metrices.items():
                if func != None:
                    score = func(pred[el_id,...],target[el_id,...])
                    if isinstance(score, Tensor):
                        score = score.to("cpu").item()
                    self.scores[name].append((score,el_name))
            self.count += 1

    def print(self):
        self._average_scores()
        output = ""
        for name, sc in self.average_scores.items():
            output += f"{name}: {sc:.5f}  "
        return output

    def get_scores(self):
        self._average_scores()
        return self.average_scores

    def get_last_scores(self):
        last_scores = {}
        for name, sc in self.scores.items():
            if len(sc) != 0:
                last_scores[str(self.prefix)+ str(name)] = sc[-1][0]
            else:
                last_scores[str(self.prefix)+ str(name)] = 0

        return last_scores

    def get_single_scores(self):
        return self.scores

    def get_single_score_per_name(self):
        output_dict = {}
        self._average_scores()
        for name, sc in self.scores.items():
            for (score, el_name) in sc:
                if el_name not in output_dict.keys():
                    output_dict[el_name] = {}
                output_dict[el_name][name] = score
        output_dict["Mean"] = self.average_scores
        return output_dict

    def _average_scores(self):
        for name, sc in self.scores.items():
            if len(sc) != 0:
                self.average_scores[str(self.prefix)+ str(name)] = sum([el[0] for el in sc])/len(sc)
            else:
                self.average_scores[str(self.prefix)+ str(name)] = 0


    def __call__(self, pred, target):
        return self.metrices[self.metrices.keys()[0]](pred, target)


class HausDorffMetric():
    def __init__(self, include_background=False, distance_metric='euclidean', percentile=None) -> None:
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        return monai.metrics.compute_hausdorff_distance(pred, target, 
            include_background=self.include_background, 
            distance_metric=self.distance_metric, 
            percentile=self.percentile)




#%%
def test_metric(metric):
    a = torch.rand((1,1,20,20,20))
    b = torch.rand((1,1,20,20,20))
    metric(a,b)


def test_struct():
    a = torch.rand((1,1,20,20,20))
    b = torch.rand((1,1,20,20,20))
    metrics = MetricesStruct({"Dice": DiceMetric()}, prefix="")
    metrics.update(a,b, el_name="num1")
    print(metrics.print())
    print(metrics.get_single_score_per_name())
    print(metrics.get_scores())
    
#test_struct()
# %%


