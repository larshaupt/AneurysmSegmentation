from math import exp
from tkinter import E
import torch.utils.data as data
import numpy as np
import os
from skimage import exposure
import pandas as pd
import pdb

class NpyDataset(data.Dataset):
    """
    Input params:
        path: Path to the folder containing the dataset (multiple npy files).
        transform: PyTorch transform to apply to every data instance (default = None).
    """
    def __init__(self, exp_config, path, transform = None):
        super().__init__()
        self.transform = transform
        self.main_path = path
        
        self.image_names = [s for s in os.listdir(path) if ('_x' in s)]
        self.image_names.sort()
        
        self.target_names = [s for s in os.listdir(path) if ('_y' in s)]
        self.target_names.sort()


    def __getitem__(self, index):
        
        patient_name_current = self.image_names[index][:-10]

        x = np.load(self.main_path + self.image_names[index], 'r')
        y = np.load(self.main_path + self.target_names[index], 'r')

        # set all values above 0 and below -250 to 0
        #x_th1 = x < 0.0
        #x_th2 = x > 250.0
        #x_th = np.logical_and(x_th1, x_th2)
        #ind_x, ind_y, ind_z = np.where(x_th == 1)
        #x[ind_x, ind_y, ind_z] = 0.0

        ct_min_current, ct_max_current = 0.0, 250.0

        #x = self.normalize_min_max(x, ct_min_current, ct_99_percentile_current)
        #y = self.normalize_min_max(y, pet_min_current, pet_99_percentile_current)

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        sample = {'image': x, 'target': y}

        if self.transform != None:
            sample = self.transform(sample)

        # print(sample['image'].shape, np.expand_dims(np.array(norm_params), 0).shape)
        # print(norm_params)
        norm_params  = np.array([])

        return sample['image'], sample['target'], norm_params, self.image_names[index]

    def normalize_min_max(self, img, min_val, max_val):
        img_normalized = (img - min_val) / (max_val - min_val)

        return img_normalized
    
    # def apply_histeq(self, x):
    #     x_histeq = exposure.equalize_hist(x)

    #     return x_histeq

    def __len__(self):
         return len(self.image_names)