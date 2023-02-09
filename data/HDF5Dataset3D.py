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
import torch
import types

class HDF5Dataset3D(data.Dataset):
    """
    The HDF5Dataset3D class inherits from the PyTorch data.Dataset class and is used for loading 3D datasets from HDF5 files. The main function of this class is the init function, which sets up the path and the transform to be applied to the data.

    init inputs:

    exp_config :unknown, not mentioned in the code
    path: path to the folder containing the dataset (multiple HDF5 files)
    data_names: list of names for images and corresponding labels, default is []
    transform: PyTorch transform to apply to every data instance, default is None
    reduce_len: used for debugging, used in len, default is -1
    mask_path: path to the folder containing mask, default is empty string ""
    norm_percentile: Percentile value for normalization, default is 99
    normalization: normalization type to be applied, can be 'minmax' or 'z' or 'zscore' default is 'minmax'

    
    The class also has two additional function

    getitem: This function is called by the PyTorch DataLoader to get a specific item at an index. It loads the image and target at the given index and returns them.
    len: returns the number of samples in the dataset.
    note:
    This class also uses pandas library for loading csv file that have statistics about the data. this file should exist in the path with the name 'file_dict.csv'
    """
    def __init__(self, 
        exp_config, path:str, data_names:list = [], transform = None, reduce_len:int = -1, mask_path:str="", norm_percentile:int = 99, normalization:str = 'minmax'):
        super().__init__()
        self.transform = transform
        self.main_path = path
        self.mask_path = mask_path
        self.reduce_len = reduce_len # for debugging, used in __len__
        self.norm_perc = norm_percentile
        self.normalization = normalization


        # For debugging
        if self.reduce_len != -1:
            print('DataLoader: Reduced Number of samples to %i'%(self.reduce_len))

        ################ Data Paths ###################
        if len(data_names) == 0:
            self.image_paths = [self.main_path + s for s in os.listdir(self.main_path) if ('_x' in s)]
            self.target_paths = [self.main_path + s for s in os.listdir(self.main_path) if ('_y' in s)]

            if self.mask_path != "":
                self.mask_paths = [self.mask_path + s for s in os.listdir(self.mask_path) if ('_mask' in s)]
            
        else:
            self.image_paths = [self.main_path + s['image']  + ".h5" for s in data_names]
            self.target_paths = [self.main_path + s['target'] + ".h5" for s in data_names]
        
            if self.mask_path != "":
                self.mask_paths =  [self.main_path + s['mask'] + ".h5" for s in data_names]



        self.image_paths.sort()
        self.target_paths.sort()
        if self.mask_path != "":
            self.mask_paths.sort()


        path_data_dict = os.path.join(path, 'file_dict.csv')
        if os.path.exists(path_data_dict):
            data_df = pd.read_csv(path_data_dict)
            self.patient_names = data_df['name']
            self.data_min, self.data_max, self.data_99_percentile = data_df['min'], data_df['max'], data_df['99per']
            if self.normalization == 'z' or self.normalization == 'zscore':
                if 'mean' in data_df.columns and 'var' in data_df.columns:
                    self.data_mean, self.data_var = data_df['mean'], data_df['var']
                else:
                    self.data_mean, self.data_var = [],[]
        else:
            print('Did not find data dict for ',path_data_dict )
            data_df = None
            self.patient_names, self.data_min, self.data_max, self.data_99_percentile = [],[],[],[],
            if self.normalization == 'z' or self.normalization == 'zscore':
                self.data_mean, self.data_var = [],[]

    def __getitem__(self, index):
        ################ Data Loading ###################
        reader_image = h5py.File(self.image_paths[index], 'r')
        reader_target = h5py.File(self.target_paths[index], 'r')
        
        x = reader_image['data'][()]
        y = reader_target['data'][()]
        if self.mask_path != "":
            reader_mask = h5py.File(self.mask_paths[index], 'r')
            mask = reader_mask['data'][()]
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = []

        patient_name_current = os.path.basename(self.image_paths[index])[:-3]
        ################ Normalization ###################
        # scales the values between 0 and 1 
        index_patient = np.where(self.patient_names == patient_name_current)[0]
        if len(index_patient) == 0:
            index_patient = -1
        else:
            index_patient = index_patient[0]

        if index_patient != -1:
            min_value = self.data_min[index_patient]
            max_value = self.data_max[index_patient]

        # Z Score Normalization
        if self.normalization == 'z' or self.normalization == 'zscore':
            if index_patient != -1:
                mean = self.data_mean[index_patient]
                var = self.data_var[index_patient]
            else:
                mean = np.mean(x)
                var = np.var(x)
            x = self.normalize_zscore(x,mean,var)
            
        # Min Max Normalization
        else:
            if self.norm_perc == 99:
                if index_patient != -1:
                    max_scale = self.data_99_percentile[index_patient]
                else:
                    max_scale = np.percentile(x, 99)

            elif self.norm_perc == 100:
                max_scale = max_value

            else:
                max_scale = np.percentile(x, self.norm_perc)
            x = self.normalize_min_max(x,min_value,max_scale)

        ################ Preprocessing ###################
        norm_params = np.array([max_value, min_value])

        x = x.astype(np.float32)
        y = y.astype(np.uint8)
        
        if x.ndim == 3:
            x = np.expand_dims(x,axis=0)
        if y.ndim == 3:
            y = np.expand_dims(y, axis=0)

        sample = {'image': x, 'target': y, 'params': norm_params, 'mask': mask, 'name': patient_name_current}

        if self.transform != None:
            # Augmentations
            sample = self.transform(sample)

        return sample['image'], sample['target'], sample['mask'], norm_params, patient_name_current

    def normalize_min_max(self, img, min_val, max_val):
        img_normalized = (img - min_val) / (max_val - min_val)

        return img_normalized

    def normalize_zscore(self, img, mean, var):
        img_normalized = (img - mean) / var

        return img_normalized

    def __len__(self):
        
        if self.reduce_len == -1:
            length = len(self.image_paths)
        else:
            length = self.reduce_len
        return length


class HDF5Dataset3D_multiple(HDF5Dataset3D):

    def __init__(self, exp_config, path, data_names = [], path_2 = None, data_names_2 = None, transform = None, reduce_len = -1, norm_percentile = 99,  normalization = 'minmax'):
        super().__init__(exp_config = exp_config, path = path, data_names = data_names,transform =  transform,reduce_len =  reduce_len, norm_percentile = norm_percentile, normalization = normalization)

        self.path_2 = path_2


        if self.reduce_len != -1:
            print('DataLoader: Reduced Number of samples to %i'%(self.reduce_len))

        
        if len(data_names_2) == 0:
            self.image_paths += [self.path_2 + s for s in os.listdir(path) if ('_x' in s)]
            self.target_paths += [self.path_2 + s for s in os.listdir(path) if ('_y' in s)]
            
        else:
            self.image_paths += [self.path_2 + s['image']  + ".h5" for s in data_names_2]
            self.target_paths += [self.path_2 + s['target'] + ".h5" for s in data_names_2]

        self.image_paths.sort()
        self.target_paths.sort()

        path_data_dict = os.path.join(self.path_2, 'file_dict.csv')
        if os.path.exists(path_data_dict):
            data_df = pd.read_csv(path_data_dict)
            self.patient_names = pd.concat((self.patient_names, data_df['name']), ignore_index=True)
            self.data_min = pd.concat((self.data_min, data_df['min']), ignore_index=True)
            self.data_max = pd.concat((self.data_max,data_df['max']), ignore_index=True)
            self.data_99_percentile = pd.concat((self.data_99_percentile, data_df['99per']), ignore_index=True)
            
        else:
            print('Did not find data dict for ',path_data_dict )
            #data_df = None
            #self.patient_names, self.data_min, self.data_max, self.data_99_percentile = [],[],[],[],
