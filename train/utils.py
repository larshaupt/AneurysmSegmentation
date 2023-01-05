# %%
import torch
import torch.nn as nn

import numpy as np
from torch import Tensor
import pandas as pd
import os
import pdb

from importlib.machinery import SourceFileLoader

from train.options import Options


# %%

# ===============
# Create directory if not exist
# ===============
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_split(path_dict:str, fold_id:int, split:str = "") -> dict|list:
    split_dict_df = pd.read_json(path_dict)
    fold_split = split_dict_df[fold_id]
    empty_dict = {'image':[],'target':[]}
    if split == "":
        fold_dict = {'train' : empty_dict, 'val': empty_dict, 'test': empty_dict}
        if 'x_train' in split_dict_df.index and 'y_train' in split_dict_df.index:
            fold_dict['train'] = [{'image':x,'target':y} for x,y in zip(fold_split['x_train'], fold_split['y_train'])]

        if 'x_test' in split_dict_df.index and 'y_test' in split_dict_df.index:
            fold_dict['test'] = [{'image':x,'target':y} for x,y in zip(fold_split['x_test'], fold_split['y_test'])]

        if 'x_val' in split_dict_df.index and 'y_val' in split_dict_df.index:
            fold_dict['val'] = [{'image':x,'target':y} for x,y in zip(fold_split['x_val'], fold_split['y_val'])]
    
    elif split == "train" or split == "val" or split == "test":
        x_name, y_name = 'x_'+split, 'y_'+split
        if x_name in split_dict_df.index and y_name in split_dict_df.index:
            fold_dict = [{'image':x,'target':y} for x,y in zip(fold_split[x_name], fold_split[y_name])]
    else:
        raise TypeError("Wrong input for split:" + str(split))
    return fold_dict


def binarize(pred, threshold=0.5, one_hot=False):
    if not isinstance(pred, Tensor):
        pred = Tensor(pred)
    
    if pred.dtype == torch.bool:
        pred = pred.int()

    orig_shape = pred.shape


    while len(pred.shape) < 5:
        pred = torch.unsqueeze(Tensor(pred), dim=0)
    # reduce everything to 4 dimensions
    batch_slices = []
    for pred_slice in pred:
        num_channels = pred_slice.shape[0]
        if num_channels == 1: # one channel
            pred_slice = pred_slice > threshold
        elif num_channels == 2: # two channels
            pred_slice = torch.unsqueeze(torch.where(pred_slice[0,...] > pred_slice[1,...], 1,0), dim=0)
        else: # more than two channels
            if not one_hot:
                pred_slice = torch.unsqueeze(torch.argmax(pred_slice, dim=0), dim=0)
            else:
                pred_slice = torch.nn.functional.one_hot(torch.argmax(pred_slice, dim=0), num_classes=num_channels).view(num_channels, *pred_slice.shape[1:])
        batch_slices.append(pred_slice)
    pred = torch.stack(batch_slices, dim=0)

    while len(pred.shape) > len(orig_shape):
        pred = torch.squeeze(pred, dim=0).int()

    return pred


        

def find_slices(target, num, ratio_pos = 0.5, target_label=5):
    if not isinstance(target, Tensor):
        target = Tensor(target)

    while len(target.shape) < 4:
        target = torch.unsqueeze(target, dim=0)
    num_neg = int(num * (1-ratio_pos))
    num_pos = num - num_neg
    (num_labels , max_x, max_y, max_z) = target.shape
    selected_slices = []
    if target_label >= num_labels:
        target_label=0
    nonzero_label = torch.nonzero(target[target_label,...], as_tuple=False)

    if len(nonzero_label) != 0:
        for index in np.linspace(0,len(nonzero_label), num_pos + 1,endpoint=False, dtype=int)[1:]: # don't take starting element
            selected_slices.append(nonzero_label[index].cpu().detach().numpy())
    if num_neg != 0:
        for index in np.linspace(0.1, 0.9, num_neg + 1, endpoint=False)[1:]:
            selected_slices.append([int(index*max_x), int(index*max_y), int(index*max_z)]) # don't take starting element

    return  selected_slices

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def make_dirs(dirs):
    for _path in dirs:
        if _path != None:
            if not os.path.exists(_path):
                os.mkdir(_path)


def load_exp_config(exp_config_path):
            
    config_module = exp_config_path.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, exp_config_path).load_module() # exp_config stores configurations in the given config file under experiments folder.

    return exp_config

def load_config(config_path, verbose=False, overwrite=None):
    if overwrite != None:
        op = Options(config_file = config_path, verbose=verbose, **overwrite)
    else:
        op = Options(config_file = config_path, verbose=verbose)
    return op.opt


def load_model(exp_config, model_weights_path): 

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError("Model weights not found. Possible weights are: "+ str(os.listdir(os.path.dirname(model_weights_path))))

    model = exp_config.model 
    model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu')))
    model.eval()

    return model



class Predictor:
    def __init__(self, exp_names:list, pre_trained_path:str , epoch = 'best_model.pth', device = 'cpu', config_overwrite:dict=None, postprocessing=False, verbose=False) -> None:
        self.exp_names = exp_names
        self.pre_trained_path = pre_trained_path
        self.epoch = epoch
        self.models = {}
        self.device = device
        self.config_overwrite = config_overwrite
        self.postprocessing = postprocessing
        self.verbose = verbose
        

        for exp_name in self.exp_names:
            self.load_model(exp_name)
        self.standard_exp_config = self.models[self.exp_names[0]][1]

    def load_model(self, exp_name):
        exp_config_path = os.path.join(self.pre_trained_path, exp_name, f'{exp_name}.json')
        exp_config = load_config(exp_config_path, verbose=self.verbose, overwrite = self.config_overwrite)
        fold_id = exp_config.fold_id
        model_weights_path = os.path.join(self.pre_trained_path,exp_name,  str(fold_id), self.epoch)
        try:
            model = load_model(exp_config, model_weights_path)
            self.models[exp_name] = (model, exp_config)

        except Exception as e:
            print(f'Could not load model for experiment {exp_name}.')
            print(e)

    def predict(self, data, exp_name, postprocessing=False):
        if not exp_name in self.models.keys():
            raise ValueError(f'Experiment {exp_name} not loaded. Please load it first.')
        (model,exp_config) = self.models[exp_name]
        with torch.no_grad():
            model = model.to(self.device)
            data = data.to(self.device)
            pred = model(data)
        if exp_config.num_classes > 1:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)

        if postprocessing:
            pred = self.postprocess(pred, data, exp_config)
        
        return pred

    def predict_all(self, data):
        preds = {}
        for exp_name in self.exp_names:
            preds[exp_name] = self.predict(data, exp_name, postprocessing=False)


    def postprocess(self, pred, data , exp_config):
        if hasattr(exp_config, "tf_post"):
            pred = pred > 0.5
            pred = binarize(pred, one_hot=True)
            sample = {'image': data.squeeze(0), 'target': pred.squeeze(0)}
            sample = exp_config.tf_post(sample)
            pred = sample['target'].unsqueeze(0)
            return pred
        raise NotImplementedError('Cannot find postprocessing function.')
            


    def predict_all_combine(self,data):
        cpu_device = torch.device('cpu')
        pred = torch.zeros_like(data, dtype=torch.float32, device=cpu_device)
        for exp_name in self.exp_names:
            pred_t = self.predict(data, exp_name, postprocessing=False).to(cpu_device)
            if pred_t.shape[1] == 3:
                pred_t = pred_t[:,2,...].unsqueeze(dim=1)
            elif pred_t.shape[1] == 22:
                pred_t = pred_t[:,4,...].unsqueeze(dim=1)
            pred += pred_t

        if self.postprocessing:
            pred = self.postprocess(pred, data, self.standard_exp_config)
        return pred

    def get_exp_config(self, exp_name):
        if not exp_name in self.models.keys():
            raise ValueError(f'Experiment {exp_name} not loaded. Please load it first.')
        return self.models[exp_name][1]

    def get_exp_names(self):
        return self.exp_names

    def get_all_exp_names(self):
        return '_'.join(self.exp_names)


    def get_all_exp_names_comp(self):
        if len(self.exp_names) > 2:
            return self.exp_names[0] + '_' + self.exp_names[1] + f'_plus_{len(self.exp_names) - 2}'
        return '_'.join(self.exp_names)