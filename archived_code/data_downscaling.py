# %%
import numpy as np
import os
import h5py
import torch
import sys
sys.path.append('../data')
from preprocessing_utils import read_data_names, animate_img
from transformations import DownsampleByScale
# %%
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_hdf5/data/'
path_to_target_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_hdf5d2/data/'
x_files,y_files = read_data_names(path_to_external_data)
# %%

scale = (2.0/3.0, 2.0/3.0, 1.0)
for x_path,y_path in zip(x_files, y_files):
    print(x_path)

    reader_image = h5py.File(os.path.join(path_to_external_data, x_path), 'r')
    reader_target = h5py.File(os.path.join(path_to_external_data, y_path), 'r')

    x = reader_image['data'][()]
    y = reader_target['data'][()]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # process x,y 

    sample = DownsampleByScale(scale)

    out = sample({'image': x.unsqueeze(0), 'target': y.unsqueeze(0)})
    
    x, y = out['image'].squeeze(0).detach().numpy(), out['target'].squeeze(0).detach().numpy()



    with h5py.File(os.path.join(path_to_target_data, x_path), 'w') as f:
        f.create_dataset('data', data=x) 

    with h5py.File(os.path.join(path_to_target_data, y_path), 'w') as f:
        f.create_dataset('data', data=y) 
# %%
