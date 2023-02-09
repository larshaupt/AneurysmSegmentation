# %%
import imp
import torch
from data import NpyDataset, HDF5Dataset3D
from data import transformations as custom_tf
from torchvision import transforms as pytorch_tf
import nibabel as nib
import pandas as pd
import os
from preprocessing import preprocessing_utils


def load_datasets(exp_config, 
            batch_size, 
            batch_size_test, 
            path_data, 
            path_data_extra = None,
            tf_train = custom_tf.ToTensor(), 
            tf_val = custom_tf.ToTensor(), 
            split_dict = None, 
            split_dict_extra = None,
            train_val_test=(), 
            reduce_number = -1, 
            num_workers=4,
            norm_percentile = 99,
            normalization:str = 'minmax'):


    train_loader, val_loader, test_loader = get_data_loaders_hdf53d(
                                            exp_config = exp_config, 
                                            batch_size = batch_size,
                                            batch_size_test = batch_size_test, 
                                            path_data = path_data, 
                                            path_data_extra = path_data_extra,
                                            tf_train = tf_train, 
                                            tf_val = tf_val, 
                                            split_dict = split_dict,
                                            split_dict_extra = split_dict_extra,
                                            train_val_test = train_val_test, 
                                            reduce_number = reduce_number, 
                                            num_workers = num_workers,
                                            norm_percentile = norm_percentile,
                                            normalization = normalization)

    return train_loader, val_loader, test_loader

def get_data_loaders_npy(exp_config, batch_size, path_train, path_test, path_val, tf_train, tf_val):
    train_loader, test_loader, val_loader = None, None, None

    if path_train != '':
        ds_train = NpyDataset.NpyDataset(exp_config, path_train, tf_train)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = exp_config.shuffle_train, num_workers = 1)
    
    if path_test != '':
        ds_test = NpyDataset.NpyDataset(exp_config, path_test, tf_val)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = exp_config.shuffle_test, num_workers = 1)
    
    if path_val != '':
        ds_validation = NpyDataset.NpyDataset(exp_config, path_val, tf_val)
        val_loader = torch.utils.data.DataLoader(ds_validation, batch_size = batch_size, shuffle = exp_config.shuffle_validation, num_workers = 1)
    
    return train_loader, val_loader, test_loader

def get_single_data_loader_hdf53d(exp_config, 
                    batch_size, 
                    path_data, 
                    tf = custom_tf.ToTensor(), 
                    data_names = None, 
                    reduce_number = -1, 
                    num_workers = 4, 
                    extra_train_set = None, 
                    extra_data_names = [],
                    norm_percentile = 99,
                    include_mask = False,
                    normalization:str = 'minmax'
                    ):

    if not os.path.exists(path_data):
        raise FileNotFoundError(f"Could not find {path_data}.")
    if include_mask:
        mask_path = path_data
    else:
        mask_path = ""
    if data_names == None:
        x,y = preprocessing_utils.read_data_names(path_data,keep_ending=False)
        data_names = [{'image': x_el, 'target': y_el} for x_el, y_el in zip(x,y)]
    if include_mask:
        for el in data_names:
            el['mask'] = el['target'].replace('_y','_mask')

    if extra_train_set != None:
        if not os.path.exists(extra_train_set):
            raise FileNotFoundError(f"Could not find {extra_train_set}.")
        if extra_data_names == None:
            x,y = preprocessing_utils.read_data_names(extra_train_set,keep_ending=False)
            extra_data_names = [{'image': x_el, 'target': y_el} for x_el, y_el in zip(x,y)]
        ds = HDF5Dataset3D.HDF5Dataset3D_multiple(
                        exp_config = exp_config, 
                        path = path_data, 
                        data_names = data_names, 
                        path_2 = extra_train_set, 
                        data_names_2 = extra_data_names, 
                        transform = tf,
                        reduce_len = reduce_number,
                        norm_percentile = norm_percentile, 
                        normalization = normalization)
    else:
        ds = HDF5Dataset3D.HDF5Dataset3D(
                        exp_config = exp_config, 
                        path = path_data, 
                        data_names = data_names, 
                        transform = tf, 
                        reduce_len = reduce_number, 
                        norm_percentile = norm_percentile, 
                        mask_path=mask_path, 
                        normalization = normalization)
    loader = torch.utils.data.DataLoader(ds, batch_size = batch_size, shuffle = True, num_workers = min(num_workers, batch_size), pin_memory = True)

    return loader


def get_data_loaders_hdf53d(exp_config, 
                batch_size, 
                batch_size_test, 
                path_data, 
                path_data_extra = None,
                tf_train = custom_tf.ToTensor(), 
                tf_val = custom_tf.ToTensor(), 
                split_dict= None,
                split_dict_extra = None, # not used
                train_val_test = (), 
                reduce_number = -1, 
                num_workers = 4,
                norm_percentile = 99,
                normalization = 'minmax',
                include_mask = False):
    


    if split_dict == None:
        train_names, val_names, test_names = None, None, None
    else:
        if len(train_val_test) == 3:
            (train_set, val_set, test_set) = train_val_test
        else:
            (train_set, val_set, test_set) = ('train', 'val', 'test')

        train_names, val_names, test_names = split_dict[train_set], split_dict[val_set], split_dict[test_set]

    
    if split_dict_extra == None:
        train_names_extra, val_names_extra, test_names_extra = None, None, None
    else:
        if len(train_val_test) == 3:
            (train_names_extra, val_names_extra, test_names_extra) = train_val_test
        else:
            (train_names_extra, val_names_extra, test_names_extra) = ('train', 'val', 'test')

        train_names, val_names, test_names = split_dict[train_set], split_dict[val_set], split_dict[test_set]

    train_loader = get_single_data_loader_hdf53d(exp_config = exp_config, 
                                    batch_size = batch_size, 
                                    path_data = path_data, 
                                    tf = tf_train, 
                                    data_names = train_names, 
                                    reduce_number = reduce_number, 
                                    num_workers = num_workers, 
                                    extra_train_set = path_data_extra, 
                                    extra_data_names = train_names_extra,
                                    normalization = normalization)


    val_loader = get_single_data_loader_hdf53d(exp_config = exp_config, 
                                    batch_size = batch_size_test, 
                                    path_data = path_data, 
                                    tf = tf_val, 
                                    data_names = val_names, 
                                    reduce_number = reduce_number, 
                                    num_workers = num_workers, 
                                    extra_train_set = None, 
                                    extra_data_names = val_names_extra,
                                    include_mask=include_mask,
                                    normalization = normalization)

    test_loader = get_single_data_loader_hdf53d(exp_config = exp_config, 
                                    batch_size = batch_size_test, 
                                    path_data = path_data, 
                                    tf = tf_val, 
                                    data_names = test_names, 
                                    reduce_number = reduce_number, 
                                    num_workers = num_workers, 
                                    extra_train_set = None, 
                                    extra_data_names = test_names_extra,
                                    include_mask=include_mask,
                                    normalization = normalization)
    
    return train_loader, val_loader, test_loader


# %%
