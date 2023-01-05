import sys
sys.path.append('../')

from data import HDF5Dataset3D
from train.utils import *
from train.options import Options


import torch
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cc3d

from scipy.stats import multivariate_normal
from nilearn.plotting import plot_anat, plot_roi


#%%
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

    labels_out = cc3d.largest_k(
            labels, k=n, 
            connectivity=26, delta=0,
            return_N=False)
    labels = labels * (labels_out > 0)

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
    labels_in = cc3d.connected_components(labels.astype(int), connectivity=26, return_N=False)
    
    stat = cc3d.statistics(labels_in)
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


def remove_small_components(labels, thr=100):
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
    labels_in = cc3d.connected_components(labels.astype(int), connectivity=26, return_N=False)
    
    stat = cc3d.statistics(labels_in)
    cc_size = stat['voxel_counts']

    vals_to_keep = np.squeeze(np.argwhere(cc_size > thr))

    inds = labels_in == vals_to_keep[:, None, None, None]
    labels[~np.any(inds, axis = 0)] = 0


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


def clip_outside_values(labels, thr=1.0):

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

    labels_in, N = cc3d.connected_components(labels, connectivity=26, return_N=True)
    stat = cc3d.statistics(labels_in)
    stat['prob'] = []
    cov = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_cov.npy')
    mean = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_mean.npy')
    shape = labels_in.shape
    loc_prob = multivariate_normal(mean=mean, cov=cov)
    
    for i, comp in enumerate(stat['centroids']):
        prob = loc_prob.pdf(normalize_loc(comp, shape))
        stat['prob'].append(prob)

    for i, prob in enumerate(stat['prob']):
        if prob < thr:
            # remove that component
            labels_in = np.where(labels_in==i, 0, labels)

    labels_in = np.reshape(labels, orig_shape)
    labels = (labels*labels_in).astype(orig_dtype)

    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels
def threshold_predictions(pred, data, thr = 1):
    mask = data > thr
    pred = pred*mask
    return pred

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

def plot_overlay(pred, data, index = -1):
    data = data.view(*data.shape[-3:])
    if isinstance(pred, torch.Tensor):
        pred = pred.view(*pred.shape[-3:])
    else:
        pred = pred.reshape(*pred.shape[-3:])
    if index == -1:
        index = data.shape[-1]//2
    plt.imshow(data[...,index], cmap='gray')
    plt.imshow(pred[...,index], cmap='jet', alpha=0.5)

def load_opt(exp_config, **kwargs):
    opt = Options(config_file=exp_config, **kwargs)
    opt_dict = opt.get_opt()
    return opt_dict

def load_data_generator(exp_config , data_names, tf_mode='test', batch_size=1):

    if tf_mode == 'train':
        tf = exp_config.tf_train
    elif tf_mode == 'val' and hasattr(exp_config, 'tf_val'):
        tf = exp_config.tf_val 
    else:
        tf = exp_config.tf_test
    ds_data = HDF5Dataset3D.HDF5Dataset3D(exp_config, exp_config.path_data, data_names, tf)
    data_loader = torch.utils.data.DataLoader(ds_data, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory=False)
    data_generator = iter(data_loader)
    return data_generator

def inference(data, model, onehot=False):
    pred = model(data)
    if pred.shape[1] > 1:
        pred = torch.softmax(pred, dim=1)
    else:
        pred = torch.sigmoid(pred)
    if not onehot:
        pred = binarize(pred, 0.5)
    pred = pred.detach().numpy()
    return pred
