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


def binarize(pred, threshold=0.5):
    if not isinstance(pred, Tensor):
        pred = Tensor(pred)
    
    if pred.dtype == torch.bool:
        pred = pred.int()

    orig_shape = pred.shape


    while len(pred.shape) < 5:
        pred = torch.unsqueeze(Tensor(pred), dim=0)

    batch_slices = []
    for pred_slice in pred:
        if pred_slice.shape[0] == 1: # one channel
            pred_slice = pred_slice > threshold
        elif pred_slice.shape[0] == 2: # two channels
            pred_slice = torch.unsqueeze(torch.where(pred_slice[0,...] > pred_slice[1,...], 1,0), dim=0)
        else: # more than two channels
            pred_slice = torch.unsqueeze(torch.argmax(pred_slice, dim=0), dim=0)
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

def load_config(config_path, verbose=False):
    op = Options(config_file = config_path, verbose=verbose)
    return op.opt


def load_model(exp_config, model_weights_path): 

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError("Model weights not found. Possible weights are: "+ str(os.listdir(os.path.dirname(model_weights_path))))

    model = exp_config.model 
    model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu')))
    model.eval()

    return model