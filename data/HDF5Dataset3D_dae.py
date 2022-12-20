from math import exp
from tkinter import E
import torch.utils.data as data
import numpy as np
import os
from skimage import exposure
import pandas as pd
import pdb
import h5py
import time

class HDF5Dataset3D_dae(data.Dataset):
    """
    Input params:
        path: Path to the folder containing the dataset (multiple npy files).
        transform: PyTorch transform to apply to every data instance (default = None).
    """
    def __init__(self, exp_config, path, data_names = [], transform = None, reduce_len = -1):
        super().__init__()
        self.transform = transform
        self.main_path = path
        self.reduce_len = reduce_len # for debugging, used in __len__


        if self.reduce_len != -1:
            print('DataLoader: Reduced Number of samples to %i'%(self.reduce_len))


        if len(data_names) == 0:
            self.image_names = [s for s in os.listdir(path) if ('_x' in s)]
            self.target_names = [s for s in os.listdir(path) if ('_y' in s)]
            
        else:
            self.image_names = [s['image'] + '.h5' for s in data_names]
            self.target_names = [s['target'] + '.h5' for s in data_names]

        self.image_names.sort()
        self.target_names.sort()

        path_data_dict = os.path.join(path, 'file_dict.csv')
        if os.path.exists(path_data_dict):
            data_df = pd.read_csv(path_data_dict)
            self.patient_names = data_df['name']
            self.data_min, self.data_max, self.data_99_percentile = data_df['min'], data_df['max'], data_df['99per']
        else:
            print('Did not find data dict for ',path_data_dict )
            data_df = None
            self.patient_names, self.data_min, self.data_max, self.data_99_percentile = [],[],[],[],

    def __getitem__(self, index):

        reader_image = h5py.File(self.main_path + self.image_names[index], 'r')

        x = reader_image['data'][()]

        patient_name_current = self.image_names[index][:-3]
        if len(self.patient_names) != 0:
            index_patient = np.where(self.patient_names == patient_name_current)[0][0]
            min_value, perc99_value = self.data_min[index_patient], self.data_99_percentile[index_patient]
        else:
            min_value, perc99_value = np.min(x), np.percentile(x, 99)
        norm_params  = np.array([min_value, perc99_value])

        # scales the values between 0 and 1 
        x = self.normalize_min_max(x,min_value,perc99_value)

        x = x.astype(np.float32)

        if x.ndim == 3:
            x = np.expand_dims(x,axis=0)


        sample = {'image': x, 'target': np.copy(x)}

        if self.transform != None:
            
            sample = self.transform(sample)

        return sample['image'], sample['target'], norm_params, patient_name_current

    def normalize_min_max(self, img, min_val, max_val):
        img_normalized = (img - min_val) / (max_val - min_val)

        return img_normalized
    
    # def apply_histeq(self, x):
    #     x_histeq = exposure.equalize_hist(x)

    #     return x_histeq

    def __len__(self):
        
        if self.reduce_len == -1:
            length = len(self.image_names)
        else:
            length = self.reduce_len
        return length

