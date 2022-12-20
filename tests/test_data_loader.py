# %%

import sys
sys.path.append('/scratch/lhauptmann/segmentation_3D')

from importlib.machinery import SourceFileLoader
from data import data_loader
from train.utils import load_split
from train.options import Options
from preprocessing.preprocessing_utils import integrate_intensity

#import matplotlib
#matplotlib.use('Agg')

import torch
from data import transformations as custom_tf
from torchvision import transforms as pytorch_tf
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/USZ_hdf5d2_sweep_1671314512/USZ_hdf5d2_sweep_1671314512.json'
options = Options(config_file=file_path)
exp_config = options.get_opt() # exp_config stores configurations in the given config file under experiments folder.

#%%
# =====================
# Define network architecture
# =====================    
#model = exp_config.model
#model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

#model.cuda()

# =========================
# Load source dataset
# =========================
split_dict = load_split(exp_config.path_split, exp_config.fold_id)
"""
source_train_loader, source_val_loader, source_test_loader = data_loader.load_datasets(exp_config,\
                                                                        1,1,\
                                                                        exp_config.path_data,\
                                                                        exp_config.tf_train,
                                                                        exp_config.tf_test,
                                                                        split_dict=split_dict, 
                                                                        num_workers= 1,
                                                                        reduce_number=10)
"""
source_loader = data_loader.get_single_data_loader_hdf53d(
    exp_config, 
    1, 
    exp_config.path_data, 
    tf = exp_config.tf_train, 
    data_names = split_dict['train'], 
    reduce_number = -1, 
    num_workers = 0)
source_iter = iter(source_loader)
# %%
data, target, norm_params, names = next(source_iter)
print(names)
#integrate_intensity(data)
plt.imshow(data[0,0,:,:,40])
# %%
def plot_files(source_iter, save_path = "/scratch/lhauptmann/segmentation_3D/data_analysis/loader_files"):
    for i in range(1):
        data, target, norm_params, names = next(source_iter)
        z_slices_with_label = torch.nonzero(target[0,...], as_tuple=False)
        if len(z_slices_with_label) != 0:
            z_slice = z_slices_with_label[int(len(z_slices_with_label)/2)][-1]
        else:
            z_slice = int(target.shape[-1]/2)
        plt.figure()
        plt.imshow(target[0,0,..., z_slice])
        plt.savefig(os.path.join(save_path, str(i) + '_target.png'))
        plt.figure()
        plt.imshow(data[0,0,..., z_slice])
        plt.savefig(os.path.join(save_path, str(i) + '_data.png'))
        plt.clf()
    #plt.figure()
    #sns.histplot(data.flatten())


#%%

def integrate_intensity(img, save_path=None):

    if isinstance(img, torch.Tensor):
        img = img.detach().numpy()

    orig_shape = img.shape


    while len(img.shape) > 3:
        img = np.squeeze(img, axis=0)
    inten_x = np.sum(img, axis=(1,2))
    inten_y = np.sum(img, axis=(0,2))
    inten_z = np.sum(img, axis=(0,1))

    plt.plot(inten_x, label='x-intensity')
    plt.plot(inten_y, label='y-intensity')
    plt.plot(inten_z, label='z-intensity')
    #plt.yscale('log')
    plt.legend()
    plt.xlabel('Pixel Index')
    plt.ylabel('Sum of Intensity')
    plt.title('Intensity sum for 3 directions')
    plt.xlim(-20,1000)
    plt.show()
    if save_path != None:
        plt.savefig(save_path)

# %%
