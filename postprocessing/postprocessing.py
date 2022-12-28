# %%
import sys
sys.path.append('../')

from data import transformations
from data import data_loader, HDF5Dataset3D
from train.utils import *
from preprocessing.preprocessing_utils import integrate_intensity, animate_img
from train.metrics import MetricesStruct

import torch
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import cc3d
from torchmetrics import Dice, Precision, Recall, AUROC
from monai.metrics import DiceMetric
from train.metrics import PrecisionMetric, RecallMetric
from scipy.stats import multivariate_normal
import scipy
from nilearn.plotting import plot_anat, plot_roi


#%%

def print_stats(pred):
    pred = pred.view(*pred.shape[-3:]).detach().numpy()
    labels_target, N_target = cc3d.connected_components(pred, connectivity=26, return_N=True)
    target_stat = cc3d.statistics(labels_target)
    print(target_stat)
    return labels_target, N_target, target_stat

def plot_l(pred):
    nifti = nib.Nifti1Image(pred.view(*pred.shape[-3:]).detach().numpy().astype(np.uint8), np.diag((*voxel_size, 1)))
    plot_anat(nifti)

def plot_l(pred):
    nifti = nib.Nifti1Image(pred.view(*pred.shape[-3:]).detach().numpy().astype(np.uint16)*32766, np.diag((*voxel_size, 1)))
    plot_anat(nifti)

def plot_r(pred, data):
    nifti = nib.Nifti1Image(pred.view(*pred.shape[-3:]).detach().numpy().astype(np.uint8), np.diag((*voxel_size, 1)))
    nifti_data = nib.Nifti1Image((data.view(*pred.shape[-3:]).detach().numpy()*32766).astype(np.uint16), np.diag((*voxel_size, 1)))
    plot_roi(nifti, nifti_data, cmap='jet')

# %%
voxel_size = (0.3,0.3,0.6)
experiment_name = 'USZ_BrainArtery_bias_sweep_1672089856'
epoch = 69
model_name = 'best_model.pth'
config_file = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/%s/USZ_BrainArtery_bias_sweep_1672089856.json' %(experiment_name)       
model_weight_path = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/%s/0/%s' %(experiment_name, model_name)
save_path ='/srv/beegfs02/scratch/brain_artery/data/training/predictions/%s/' %(experiment_name)
img=True 
gifs = False 
nifti=True
split = 'test' 
num_slices = 5
num=10
tf_mode='test'
scale_factor = (10/3, 10/3, 5/3)
save=False

#%%
exp_config = load_config(config_file)
model = load_model(exp_config, model_weight_path)
data_names = load_split(exp_config.path_split.replace("_downscaled", "d"), exp_config.fold_id, split=split)

if tf_mode == 'train':
    tf = exp_config.tf_train
elif tf_mode == 'val' and hasattr(exp_config, 'tf_val'):
    tf = exp_config.tf_val 
else:
    tf = exp_config.tf_test
path_to_gifs, path_to_imgs, path_to_nifti = None, None, None
ds_data = HDF5Dataset3D.HDF5Dataset3D(exp_config, exp_config.path_data, data_names, tf)
data_loader = torch.utils.data.DataLoader(ds_data, batch_size = 1, shuffle = False, num_workers = 0, pin_memory=False)
data_generator = iter(data_loader)

#%%

data, target, mask,norm_params, name = next(data_generator)

# %%
pred = model(data)

if exp_config.num_classes > 1:
    pred = torch.softmax(pred, dim=1)
else:
    pred = torch.sigmoid(pred)
pred = binarize(pred, 0.5)
pred = pred.detach().numpy()[0,0,...]
# %%

pred = scipy.ndimage.binary_opening(pred, iterations=2).astype(np.uint8)
pred = scipy.ndimage.binary_closing(pred, iterations=20).astype(np.uint8)

# %%

pred = take_n_largest_components(pred, n=1)
#pred = remove_n_largest_components(pred, n=1)
#pred = clip_outside_values(pred, thr=1)


#%%
pred_t = torch.Tensor(pred).view(1,1,*pred.shape[-3:])
metric = MetricesStruct(exp_config.criterion_metric, prefix='')
metric.update(binarize(pred_t), target, el_name=name[0])
scores = metric.get_single_score_per_name()
metric.print()
# %%
pred = binarize(pred)
target = binarize(target)
pred = pred.detach().numpy()[0,0,...]
target = target.detach().numpy()[0,0,...]
data = data.detach().numpy()[0,0,...]
#%%
import cc3d

labels_pred, N_pred = cc3d.largest_k(
  pred, k=10, 
  connectivity=26, delta=0,
  return_N=True,
)
labels_target, N_target = cc3d.connected_components(target, connectivity=26, return_N=True)

pred_stat = cc3d.statistics(labels_pred)
target_stat = cc3d.statistics(labels_target)

print_pretty = lambda x: pd.DataFrame({key:[str(el) for el in x[key]] for key in x.keys()})

# %%


def normalize_loc(loc, shape):
    loc_new = loc.copy()
    for i in range(len(loc_new)):
        loc_new[i] = float(loc_new[i])/shape[i]
    return loc_new
normalize_loc(pred_stat['centroids'][0], target.shape)
# %%

for label, image in cc3d.each(labels_target, binary=False, in_place=True):
    z_slices = np.nonzero(image)
    z_slice = z_slices[-1][int(len(z_slices[2])/2)]
    plt.imshow(image[:,:,z_slice]) # stand in for whatever you'd like to do

#%%
from scipy.stats import multivariate_normal
cov = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_cov.npy')
mean = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_mean.npy')
shape = target.shape
loc_prob = multivariate_normal(mean=mean, cov=cov)
for comp in target_stat['centroids']:
    print(comp,normalize_loc(comp, shape),  loc_prob.pdf(normalize_loc(comp, shape)))


#%%

for i, (data, target, mask,norm_params, name) in enumerate(data_loader):
    if i >= num and num!=-1:
        break
    
    pred = model(data)
    pred = torch.sigmoid(pred)
    pred = binarize(pred)
    target = binarize(target)
    pred = take_n_largest_components(pred, n=15)
    pred = clip_outside_values(pred, thr=0.1)
    metric.update(pred, target, el_name=name[0])


scores = metric.get_single_score_per_name()
print(f"Scores for {experiment_name} with {tf_mode} transform on {split} split:")
scores_df = pd.DataFrame(data=scores).transpose()
print(scores_df)

# %%
def take_n_largest_components(labels, n=5):
    orig_shape = labels.shape
    if isinstance(labels, torch.Tensor):
        torch_flag = True
        labels = labels.detach().numpy()
    else:
        torch_flag = False
    assert labels.ndim in [3,4,5], 'wrong input shape, too many or too few dimensions'
    assert issubclass(labels.dtype.type, np.integer) or issubclass(labels.dtype.type, np.bool8), f'Wrong input type. Need to be int. Found: {labels.dtype}'
    orig_shape = labels.shape
    orig_dtype = labels.dtype

    labels = np.reshape(labels, labels.shape[-3:])

    labels = cc3d.largest_k(
            labels, k=n, 
            connectivity=26, delta=0,
            return_N=False)
    labels = np.where(labels > 0, 1,0)

    labels = np.reshape(labels, orig_shape)
    labels = labels.astype(orig_dtype)

    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels

def remove_n_largest_components(labels, n=5):
    orig_shape = labels.shape
    if isinstance(labels, torch.Tensor):
        torch_flag = True
        labels = labels.detach().numpy()
    else:
        torch_flag = False
    assert labels.ndim in [3,4,5], 'wrong input shape, too many or too few dimensions'
    assert issubclass(labels.dtype.type, np.integer) or issubclass(labels.dtype.type, np.bool8), f'Wrong input type. Need to be int. Found: {labels.dtype}'
    orig_shape = labels.shape
    orig_dtype = labels.dtype

    labels = np.reshape(labels, labels.shape[-3:])
    labels_in = cc3d.connected_components(labels, connectivity=26, return_N=False)
    
    stat = cc3d.statistics(labels)
    cc_size = stat['voxel_counts']
    largest_c = np.argsort(cc_size)

    vals_to_keep = largest_c[:max(len(largest_c)-n,0)]
    inds = labels == vals_to_keep[:, None, None, None]
    labels[~np.any(inds, axis = 0)] = 0

    labels = np.where(labels > 0, 1,0)

    labels = np.reshape(labels, orig_shape)
    labels = labels.astype(orig_dtype)

    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels

def normalize_loc(loc, shape):
    loc_new = loc.copy()
    for i in range(len(loc_new)):
        loc_new[i] = float(loc_new[i])/shape[i]
    return loc_new


def take_n_prob_comp(labels, n=5):

    orig_shape = labels.shape
    if isinstance(labels, torch.Tensor):
        torch_flag = True
        labels = labels.detach().numpy()
    else:
        torch_flag = False
    assert labels.ndim in [3,4,5], 'wrong input shape, too many or too few dimensions'
    assert issubclass(labels.dtype.type, np.integer) or issubclass(labels.dtype.type, np.bool8), f'Wrong input type. Need to be int. Found: {labels.dtype}'
    orig_shape = labels.shape
    orig_dtype = labels.dtype

    labels = np.reshape(labels, labels.shape[-3:])

    labels, N = cc3d.connected_components(labels, connectivity=26, return_N=True)
    stat = cc3d.statistics(labels)
    stat['prob'] = []
    cov = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_cov.npy')
    mean = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_mean.npy')
    shape = labels.shape
    loc_prob = multivariate_normal(mean=mean, cov=cov)
    
    for i, comp in enumerate(stat['centroids']):
        prob = loc_prob.pdf(normalize_loc(comp, shape))
        stat['prob'].append(prob)

    print(stat)
    indices = np.argsort(stat['prob'])[::-1]
    indices = indices[:min(N, n)]
    labels_new = np.zeros_like(labels)
    for ind in indices:
        labels_new =labels_new +  np.where(labels ==ind, 1,0)

    labels = labels_new

    labels = np.reshape(labels, orig_shape)
    labels = labels.astype(orig_dtype)

    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels


def clip_outside_values(labels, thr=1):

    orig_shape = labels.shape
    if isinstance(labels, torch.Tensor):
        torch_flag = True
        labels = labels.detach().numpy()
    else:
        torch_flag = False
    assert labels.ndim in [3,4,5], 'wrong input shape, too many or too few dimensions'
    assert issubclass(labels.dtype.type, np.integer) or issubclass(labels.dtype.type, np.bool8), f'Wrong input type. Need to be int. Found: {labels.dtype}'
    orig_shape = labels.shape
    orig_dtype = labels.dtype

    labels = np.reshape(labels, labels.shape[-3:])

    labels, N = cc3d.connected_components(labels, connectivity=26, return_N=True)
    stat = cc3d.statistics(labels)
    stat['prob'] = []
    cov = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_cov.npy')
    mean = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_mean.npy')
    shape = labels.shape
    loc_prob = multivariate_normal(mean=mean, cov=cov)
    
    for i, comp in enumerate(stat['centroids']):
        prob = loc_prob.pdf(normalize_loc(comp, shape))
        stat['prob'].append(prob)

    for i, prob in enumerate(stat['prob']):
        if prob < thr:
            labels = np.where(labels==i, 0, labels)

    labels = np.reshape(labels, orig_shape)
    labels = labels.astype(orig_dtype)

    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels


# %%
a = clip_outside_values(pred, thr =1)
# %%

metric_w = MetricesStruct(exp_config.criterion_metric, prefix='')
metric_wo = MetricesStruct(exp_config.criterion_metric, prefix='')
for data, target, norm_params, name in data_loader:


    pred = model(data)

    if exp_config.num_classes > 1:
        pred = torch.softmax(pred, dim=1)
    else:
        pred = torch.sigmoid(pred)
    pred = binarize(pred, 0.5)


    metric_wo.update(binarize(pred), target, el_name=name[0])

    pred = take_n_largest_components(pred, n=10)
    pred = clip_outside_values(pred, thr=1)


    metric_w.update(binarize(pred), target, el_name=name[0])


# %%
scores = metric_wo.get_single_score_per_name()
print(f"Scores for {experiment_name} with {tf_mode} transform on {split} split:")
scores_df = pd.DataFrame(data=scores).transpose()
print(scores_df)
# %%
