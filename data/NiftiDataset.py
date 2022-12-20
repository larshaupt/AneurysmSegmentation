from math import exp
import torch.utils.data as data
import h5py
import numpy as np
import os
from skimage import exposure
import nibabel as nib
import pdb
import time
import matplotlib.pyplot as plt
import pandas as pd
class NiftiDataset(data.Dataset):
    """
    Input params:
        path: Path to the folder containing the dataset (one or multiple HDF5 files).
        transform: PyTorch transform to apply to every data instance (default = None).
    """
    def __init__(self, exp_config, path, transform = None):
        super().__init__()
        self.transform = transform
        self.main_path = path
        
        self.ct_names = [s for s in os.listdir(path) if ('_CT_' in s)]
        self.ct_names.sort()
        
        self.pet_names = [s for s in os.listdir(path) if ('_PET_' in s)]
        self.pet_names.sort()
        
        self.seg_names1 = [s for s in os.listdir(path) if ('_SEG_' in s) and ('_-250_-50_2.nii.gz' in s)]
        self.seg_names1.sort()
        
        self.seg_names2 = [s for s in os.listdir(path) if ('_SEG_' in s) and ('_-190_-10_1.nii.gz' in s)]
        self.seg_names2.sort()

        ind_search = path.find('folds')
        ind_slash_rest = path[ind_search:].find('/')
        path_folds = path[:ind_search + ind_slash_rest]

        path_ct = '%s/ct_min_max_percentile.csv'%(path_folds)
        path_pet = '%s/pet_min_max_percentile.csv'%(path_folds)
        df_ct = pd.read_csv(path_ct)
        df_pet = pd.read_csv(path_pet)

        self.patient_names = df_ct['name']
        self.ct_min, self.ct_max, self.ct_99_percentile = df_ct['min'], df_ct['max'], df_ct['99_percentile']
        self.pet_min, self.pet_max, self.pet_99_percentile = df_pet['min'], df_pet['max'], df_pet['99_percentile']

                    
    def __getitem__(self, index):
        ct = nib.load('%s/%s'%(self.main_path, self.ct_names[index])).get_fdata()
        pet = nib.load('%s/%s'%(self.main_path, self.pet_names[index])).get_fdata()
        seg = nib.load('%s/%s'%(self.main_path, self.seg_names1[index])).get_fdata()
        
        try:
            patient_name_current = int(self.ct_names[index][:-14])
        except ValueError:
            patient_name_current = self.ct_names[index][:-14]

        index_patient = np.where(self.patient_names == patient_name_current)[0][0]
        ct_min_current, ct_max_current, ct_99_percentile_current = self.ct_min[index_patient], self.ct_max[index_patient], self.ct_99_percentile[index_patient]
        pet_min_current, pet_max_current, pet_99_percentile_current = np.min(self.pet_min), np.max(self.pet_max), np.percentile(self.pet_max, 99)

        ct = self.normalize_min_max(ct, ct_min_current, ct_99_percentile_current)
        pet = self.normalize_min_max(pet, pet_min_current, pet_99_percentile_current)

        ct = ct.astype(np.float32)
        pet = pet.astype(np.float32)
        seg = seg.astype(np.float32)

        sample = {'image': ct, 'target': pet, 'seg': seg}

        if self.transform != None:
            sample = self.transform(sample)

        norm_params = np.array([ct_min_current, ct_99_percentile_current, pet_min_current, pet_99_percentile_current])

        return sample['image'], sample['target'], sample['seg'], norm_params, patient_name_current

    def normalize_min_max(self, img, min_val, max_val):
        img_normalized = (img - min_val) / (max_val - min_val)

        return img_normalized

    def __len__(self):
         return len(self.ct_names)