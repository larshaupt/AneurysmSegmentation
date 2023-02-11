# %%
import imp
import torch
from data import NpyDataset, HDF5Dataset3D_dae
from data import transformations as custom_tf
from torchvision import transforms as pytorch_tf
import nibabel as nib
import pandas as pd
import os
from preprocessing import preprocessing_utils


def load_datasets(exp_config, batch_size, batch_size_test, path_data, tf_train = custom_tf.ToTensor(), tf_val = custom_tf.ToTensor(), split_dict = None, train_val_test=(), reduce_number = -1, num_workers=4):
    
    train_loader, val_loader, test_loader = get_data_loaders_hdf53d_dae(exp_config, batch_size,batch_size_test, path_data, tf_train, tf_val, split_dict,train_val_test, reduce_number, num_workers)
    return train_loader, val_loader, test_loader


def get_single_data_loader_hdf53d_dae(exp_config, batch_size, path_data, tf = custom_tf.ToTensor(), data_names = None, reduce_number = -1, num_workers = 4):

    if data_names == None:
        x,y = preprocessing_utils.read_data_names(path_data,keep_ending=False)
        data_names = [{'image': x_el, 'target': y_el} for x_el, y_el in zip(x,y)]

    ds = HDF5Dataset3D_dae.HDF5Dataset3D_dae(exp_config, path_data, data_names, tf, reduce_number)
    loader = torch.utils.data.DataLoader(ds, batch_size = batch_size, shuffle = True, num_workers = min(num_workers, batch_size), pin_memory = False)

    return loader

def get_data_loaders_hdf53d_dae(exp_config, batch_size, batch_size_test, path_data, tf_train = custom_tf.ToTensor(), tf_val = custom_tf.ToTensor(), split_dict= None, train_val_test = (), reduce_number = -1, num_workers = 4):
    
    if split_dict == None:
        train_names, val_names, test_names = None, None, None

    else:
        if len(train_val_test) == 3:
            (train_set, val_set, test_set) = train_val_test
        else:
            (train_set, val_set, test_set) = ('train', 'val', 'test')
        train_names, val_names, test_names = split_dict[train_set], split_dict[val_set], split_dict[test_set]

    train_loader = get_single_data_loader_hdf53d_dae(exp_config, batch_size, path_data, tf_train, train_names, reduce_number, num_workers)
    val_loader = get_single_data_loader_hdf53d_dae(exp_config, batch_size_test, path_data, tf_val, val_names, reduce_number, num_workers)
    test_loader = get_single_data_loader_hdf53d_dae(exp_config, batch_size_test, path_data, tf_val, test_names, reduce_number, num_workers)
    
    return train_loader, val_loader, test_loader


# %%
