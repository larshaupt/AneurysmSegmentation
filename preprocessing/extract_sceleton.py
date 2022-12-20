#%%
import torch
import numpy as np
import sys
sys.path.append('/scratch/lhauptmann/segmentation_3D')

from importlib.machinery import SourceFileLoader
from data import data_loader
from train.utils import *
from preprocessing_utils import read_data_names, animate_img
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from skimage.morphology import binary_closing, ball, skeletonize
import plotly.graph_objects as go
import cc3d


#%%
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_hdf5/data/'
x_files,y_files = read_data_names(path_to_external_data)

#%%
index=0
reader_image = h5py.File(os.path.join(path_to_external_data, x_files[index]))
reader_target = h5py.File(os.path.join(path_to_external_data, y_files[index]))

x = reader_image['data'][()]
y = reader_target['data'][()]
x = (x-x.min())/(x.max()-x.min())
y = (y-y.min())/(y.max()-y.min())
# %%
def plot_data(data, target):
    slices_ind = find_slices(target, 1, ratio_pos=1.0)
    print(slices_ind)
    plt.imshow(data[:,:,slices_ind[0][-1]]/data.max())
# %%
x_ang = binary_closing(x>0.25, ball(6))

# %%
labels_out, N = cc3d.connected_components(x_ang, connectivity=6, return_N=True)
stats = cc3d.statistics(labels_out)
labels = np.intersect1d(np.argwhere(stats['voxel_counts'] > 100), np.argwhere(stats['voxel_counts'] < 10000))
labels_out = labels_out 
# %%
center = np.median(np.argwhere(x>0.3), axis=0).astype("int")
plt.imshow(x[:,:, center[2]])
plt.scatter(center[1], center[0], color= "red")
# %%
