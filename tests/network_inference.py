# %%
import sys
sys.path.append('/srv/beegfs02/scratch/brain_artery/data/Segmentation3D')

import train.utils as ut
from data import transformations, data_loader, HDF5Dataset3D

from preprocessing.preprocessing_utils import integrate_intensity, animate_img, save_to_nifti
from train.metrics import MetricesStruct

import pandas as pd
import torch
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from importlib.machinery import SourceFileLoader
import os
import numpy as np
from test_utils import save_pred, make_predictions



#%%
#experiment_names = ['USZ_BrainArtery_bias_sweep_1672294348', 'USZ_BrainArtery_bias_sweep_1672266130']
experiment_names_adam= ['Adam_sweep_1672475897', 'Adam_sweep_1672501329', 'Adam_sweep_1672475897','Adam_sweep_1672488572', 'Adam_sweep_1672489120']
experiment_names = ['USZ_BrainArtery_bias_sweep_1672678241']
experiment_names_best_bce = ["USZ_BrainArtery_bias_sweep_1672864585", "USZ_BrainArtery_bias_sweep_1672860029", "USZ_BrainArtery_bias_sweep_1672859412", "USZ_BrainArtery_bias_sweep_1672846254","USZ_BrainArtery_bias_sweep_1672846264" ] #best model run
experiment_names_best_softdice = ["USZ_BrainArtery_bias_sweep_1672860983", "USZ_BrainArtery_bias_sweep_1672859830", "USZ_BrainArtery_bias_sweep_1672849038", "USZ_BrainArtery_bias_sweep_1672846264","USZ_BrainArtery_bias_sweep_1672846264" ] #best model run
experiment_names_best_translr = ['USZ_BrainArtery_bias_sweep_1672918520', 'USZ_BrainArtery_bias_sweep_1672914714', 'USZ_BrainArtery_bias_sweep_1672914729', 'USZ_BrainArtery_bias_sweep_1672914739', 'USZ_BrainArtery_bias_sweep_1672914713']
experiment_names_wholevol22 = ['USZ_BrainArtery_bias111_sweep_1673034756', 'USZ_BrainArtery_bias111_sweep_1673037679', 'USZ_BrainArtery_bias111_sweep_1673018837', 'USZ_BrainArtery_bias111_sweep_1673000985', 'USZ_BrainArtery_bias111_sweep_1672981931']
experiment_names_wholevol3 = ['USZ_BrainArtery_bias111_sweep_1673023809','USZ_BrainArtery_bias111_sweep_1673007015','USZ_BrainArtery_bias111_sweep_1672989580','USZ_BrainArtery_bias111_sweep_1672970792'] #USZ_BrainArtery_bias111_sweep_1673034757

experiment_names_wholevol1 = ['USZ_BrainArtery_bias111_sweep_1673034941','USZ_BrainArtery_bias111_sweep_1673034247','USZ_BrainArtery_bias111_sweep_1673013749','USZ_BrainArtery_bias111_sweep_1672995677','USZ_BrainArtery_bias111_sweep_1672977086']
experiment_names_wholevoldice3 = ['USZ_BrainArtery_bias111_sweep_1673135808_4',
 'USZ_BrainArtery_bias111_sweep_1673125252_3',
 'USZ_BrainArtery_bias111_sweep_1673122100_2',
 'USZ_BrainArtery_bias111_sweep_1673109693_1',
 'USZ_BrainArtery_bias111_sweep_1673109692_0']
experiment_names_wholevoldice22 = ['USZ_BrainArtery_bias111_sweep_1673135808_4',
 'USZ_BrainArtery_bias111_sweep_1673125252_3',
 'USZ_BrainArtery_bias111_sweep_1673122100_2',
 'USZ_BrainArtery_bias111_sweep_1673109693_1',
 'USZ_BrainArtery_bias111_sweep_1673109692_0']

experiment_names_wholevoldice1 = ['USZ_BrainArtery_bias111_sweep_1673136630_4',
 'USZ_BrainArtery_bias111_sweep_1673132155_3',
 'USZ_BrainArtery_bias111_sweep_1673122432_2',
 'USZ_BrainArtery_bias111_sweep_1673109692_0',
 'USZ_BrainArtery_bias111_sweep_1673109704_1']

experiment_names_patchbce3  = ['USZ_BrainArtery_bias_sweep_1672465444_4',
 'USZ_BrainArtery_bias_sweep_1672424097_3',
 'USZ_BrainArtery_bias_sweep_1672364412_2',
 'USZ_BrainArtery_bias_sweep_1672308271_1',
 'USZ_BrainArtery_bias_sweep_1672252138_0']

experiment_names_patchbce1  = ['USZ_BrainArtery_bias_sweep_1672476739_4',
 'USZ_BrainArtery_bias_sweep_1672437140_3',
 'USZ_BrainArtery_bias_sweep_1672373421_2',
 'USZ_BrainArtery_bias_sweep_1672321174_1',
 'USZ_BrainArtery_bias_sweep_1672266130_0']

experiment_names_mixtrain = ['USZ_BrainArtery_bias_sweep_1673066297_4',
 'USZ_BrainArtery_bias_sweep_1673046784_3',
 'USZ_BrainArtery_bias_sweep_1672970716_1',
 'USZ_BrainArtery_bias_sweep_1672970716_2',
 'USZ_BrainArtery_bias_sweep_1672970713_0']
experiment_names = experiment_names_patchbce1

experiment_names = [exp_n if exp_n[-2] != '_' else exp_n[:-2] for exp_n in experiment_names]

model_name = 'best_model.pth'
pre_trained_path = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained'
save_path = '/srv/beegfs02/scratch/brain_artery/data/training/predictions/'

config_overwrite = {'val_threshold_data': 0.8, "apply_mask": False, "val_threshold_cc": 50, "val_threshold_cc_max": 10000, "crop_sides": True, 'compute_mdice': True, "add_own_hausdorff": True}

make_predictions( 
            experiment_names,
            pre_trained_path, 
            epoch = model_name,
            save_path = save_path, 
            img=True,  
            gifs = False, 
            nifti=True, 
            split = 'test', 
            num_slices = 5, 
            num=-1 ,
            tf_mode='val', 
            voxel_size = (0.3,0.3,0.6),
            save = False,
            binarize_target = True,
            config_overwrite = config_overwrite,
            postprocessing= False,
            postfix = "_bestbce",
            aggregation_strategy = "mean",
            factor=1.0,
            softmax=False)  

input_dict = {"experiment_names":experiment_names,
            "pre_trained_path":pre_trained_path,
            "epoch":model_name,
            "save_path":save_path,
            "img":True,
            "gifs":False,
            "nifti":True,
            "split":'test',
            "num_slices": 5,
            "num":-1,
            "tf_mode":'test',
            "voxel_size":(0.3, 0.3, 6.0),
            "save":False,
            "binarize_target":True,
            "config_overwrite":config_overwrite,
            "postprocessing":True,
            "postfix":"_alladam_wp",
            "softmax": "max"
            }



run_params = [{"experiment_names":experiment_names_best_softdice + experiment_names_best_bce,
            "postprocessing":True,
            "factor":1.0,
            "aggregation_strategy":"mean",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_all_wp_1"},
            {"experiment_names":experiment_names_best_softdice + experiment_names_best_bce,
            "postprocessing":True,
            "factor":2.0,
            "aggregation_strategy":"mean",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_all_wp_2"},
            {"experiment_names":experiment_names_best_softdice + experiment_names_best_bce,
            "postprocessing":True,
            "factor":3.0,
            "aggregation_strategy":"mean",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_all_wp_3"},
            {"experiment_names":experiment_names_best_softdice + experiment_names_best_bce + experiment_names_best_translr,
            "postprocessing":True,
            "factor":1.0,
            "aggregation_strategy":"mean",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_alltl_wp_1",},
            {"experiment_names": experiment_names_best_translr,
            "postprocessing":True,
            "factor":1.0,
            "aggregation_strategy":"mean",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_tl_wp_1",},
            {"experiment_names": experiment_names_best_translr,
            "postprocessing":True,
            "factor":1.0,
            "aggregation_strategy":"max",
            "config_overwrite":{'val_threshold_data': 0.8, "apply_mask": True, "val_threshold_cc": 100, "val_threshold_cc_max": 5000, "crop_sides": True},
            "postfix":"_tl_wp_max",}
            ]
# %%
def run_sweep():
    for run_param in run_params:
        print(run_param)
        input_dict.update(run_param)
        print(input_dict)
        make_predictions(**input_dict)

#run_sweep()  

""" make_predictions( 
            experiment_names,
            pre_trained_path, 
            epoch = model_name,
            save_path = save_path, 
            img=True,  
            gifs = False, 
            nifti=True, 
            split = 'test', 
            num_slices = 5, 
            num=-1 ,
            tf_mode='test', 
            scale_factor = (1.0, 1.0, 1.0),
            save = True,
            binarize_target = True,
            config_overwrite = config_overwrite,
            postprocessing= True,
            postfix = "_alladam_wp") """


# %%