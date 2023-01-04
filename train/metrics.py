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
import time
import SimpleITK as sitk
import scipy

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
            return np.NAN
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
            return np.NaN
            
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
            return np.NaN
            
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
        if pred.shape[0] != 1:
            pred = pred[self.target_class,:,:,:].unsqueeze(0)

        if target.shape[0] != 1:
            target = target[self.target_class,:,:,:].unsqueeze(0)

        return self.metric(pred, target)


class MetricesStruct():
    def __init__(self, metrices, prefix:str = "", debug = False) -> None:

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
        self.debug = debug

    def update(self, pred, target, el_name=""):

        for el_id in range(pred.shape[0]): 
            for name, func in self.metrices.items():
                if func != None:
                    if self.debug:
                        start_time = time.time()
                    score = func(pred[el_id,...],target[el_id,...])
                    if self.debug:
                        print(f'Metric computing time for {name}: {round(time.time()-start_time, 4)} seconds')
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


class HausDorffMetricMonai():
    def __init__(self, include_background=False, distance_metric='euclidean', percentile=None) -> None:
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        return monai.metrics.compute_hausdorff_distance(pred, target, 
            include_background=self.include_background, 
            distance_metric=self.distance_metric, 
            percentile=self.percentile)


class HausDorffMetric():
    def __init__(self, percentile=95) -> None:
        self.percentile = percentile
    def __call__(self, pred: Tensor, target: Tensor):
        #result_statistics = sitk.StatisticsImageFilter()
        #result_statistics.Execute(result_image)
        pred = pred.detach().numpy()
        target = target.detach().numpy()
        pred_sum = np.sum(pred)
        if pred_sum == 0:
            hd = torch.nan
            return hd

        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
        #e_test_image = sitk.BinaryErode(test_image, (1, 1, 1))
        #e_result_image = sitk.BinaryErode(result_image, (1, 1, 1))
        e_target = scipy.ndimage.binary_erosion(target)
        e_pred = scipy.ndimage.binary_erosion(pred)

        #h_test_image = sitk.Subtract(test_image, e_test_image)
        #h_result_image = sitk.Subtract(result_image, e_result_image)
        h_target = target - e_target
        h_pred = pred- e_pred


        h_target_indices = np.argwhere(h_target)
        h_pred_indices = np.argwhere(h_pred)

        #h_test_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_test_image))).tolist()
        #h_result_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_result_image))).tolist()

        target_coordinates = h_target_indices
        pred_coordinates = h_pred_indices
        #test_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_test_indices]
        #result_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_result_indices]
        
        def get_distances_from_a_to_b(a, b):
            kd_tree = scipy.spatial.KDTree(a, leafsize=100)
            return kd_tree.query(b, k=1, eps=0, p=2)[0]

        d_test_to_result = get_distances_from_a_to_b(target_coordinates, pred_coordinates)
        d_result_to_test = get_distances_from_a_to_b(pred_coordinates, target_coordinates)

        hd = max(np.percentile(d_test_to_result, self.percentile), np.percentile(d_result_to_test, self.percentile))
        
        return hd

class VolumetricSimilarityMetric():
    def __init__(self) -> None:
        pass
    def __call__(self, pred: Tensor, target: Tensor) -> None:

        """
        Volumetric Similarity.
        
        VS = 1 -abs(A-B)/(A+B)
        
        A = ground truth
        B = predicted     
        """
        
        #test_statistics = sitk.StatisticsImageFilter()
        #result_statistics = sitk.StatisticsImageFilter()
        
        #test_statistics.Execute(target)
        #result_statistics.Execute(pred)
        
        #numerator = abs(test_statistics.GetSum() - result_statistics.GetSum())
        #denominator = test_statistics.GetSum() + result_statistics.GetSum()

        pred, target = convert_to_tensor(pred), convert_to_tensor(target)
        assert pred.shape[0] == 1 and target.shape[0] == 1 
        pred, target = ut.binarize(pred), ut.binarize(target)

        
        FN = (torch.where(pred==0, 1,0)*target).sum()
        FP = (torch.where(target==0, 1,0)*pred).sum()
        TP = (pred*target).sum()

        numerator = abs(FN - FP)
        denominator = 2*TP + FP + FN
        if denominator > 0:
            vs = 1 - (numerator / denominator)
        else:
            vs = torch.nan
             
        return vs


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


