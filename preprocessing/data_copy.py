#%%
import numpy as np # data handling
import os # path handling
import matplotlib.pyplot as plt # visualization
import nibabel as nib # mri nii file reading 
import pandas as pd # for general table handling
import pyvista as pv # for reading vtk
import seaborn as sns # for beautiful plots
import shutil
import h5py
import pandas as pd
from preprocessing_utils import correct_labels

import sys
sys.path.append('/scratch/lhauptmann/BrainSeg/src/utils')
from preprocessing_utils import read_data_names

# Write all important paths for read/save here
# if you don't know them, just ommit them (might evoke some errors)

path_to_data = "../data/USZ_BrainArtery/"
path_to_processed_data = "../data/USZ_BrainArtery_processed/"
path_to_processed_bias_data = "../data/USZ_BrainArtery_processed_bias/"
path_to_class_labels_table = "../data/USZ_BrainArtery/class_labels.xlsx"
path_to_headers = "../data/USZ_BrainArtery_headers/"
path_to_data_analysis = "../data_analysis/"
path_to_raw_data = "../data/USZ_BrainArtery_raw/"
path_to_data_splits = "data_splits/"
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ/data'
path_to_external_data_hd5f = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_hdf5/data'
path_to_new_data = "//usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/ADAM/data"
# %%

# reads in volume data


def create_fold_copies(dict_path, data_path, target_path):

    k_fold_dict = pd.read_json(dict_path)


    for fold in [0]: #range(k_fold_dict.shape[1]):
        print('Processing fold ', fold)
        fold_dir = os.path.join(target_path, str(fold))
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        
        for dir in ['train/', 'test/', 'val/']:
            path_dir = os.path.join(fold_dir, dir)
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)
        print('Test')
        for i,x in enumerate(k_fold_dict[fold]['x_test']):
            y = k_fold_dict[fold]['y_test'][i]
            #shutil.copy(os.path.join(data_path, x[:-4] + '.h5'), os.path.join(fold_dir,'test/', x[:-4] + '.h5'))
            #shutil.copy(os.path.join(data_path, y[:-4] + '.h5'), os.path.join(fold_dir,'test/', y[:-4] + '.h5'))
        print('Train')
        for i,x in enumerate(k_fold_dict[fold]['x_train']):
            y = k_fold_dict[fold]['y_train'][i]
            #shutil.copy(os.path.join(data_path, x[:-4] + '.h5'), os.path.join(fold_dir,'train/', x[:-4] + '.h5'))
            #shutil.copy(os.path.join(data_path, y[:-4] + '.h5'), os.path.join(fold_dir,'train/', y[:-4] + '.h5'))
        print('Val')
        for i,x in enumerate(k_fold_dict[fold]['x_val']):
            y = k_fold_dict[fold]['y_val'][i]
            shutil.copy(os.path.join(data_path, x[:-4] + '.h5'), os.path.join(fold_dir,'val/', x[:-4] + '.h5'))
            shutil.copy(os.path.join(data_path, y[:-4] + '.h5'), os.path.join(fold_dir,'val/', y[:-4] + '.h5'))

_target_path = os.path.join(path_to_external_data_hd5f,'folds/')
_dict_path = os.path.join(path_to_data_splits, 'k_fold_split2_val.json')
_data_path = os.path.join(path_to_external_data_hd5f,'data/')
#create_fold_copies(_dict_path, _data_path, _target_path)

#%%

def convert_npy_2_hdf5(data_path, target_path):
    x,y = read_data_names(data_path)
    for i,x_file in enumerate(x):
        y_file = y[i]
        print('Number ', str(i), '   File ', x_file)
        x_data = np.load(os.path.join(data_path, x_file))
        with h5py.File(os.path.join(target_path, x_file[:-4] + '.h5'), 'w') as f:
            f.create_dataset('data', data=x_data) 

        y_data = np.load(os.path.join(data_path, y_file))
        with h5py.File(os.path.join(target_path, y_file[:-4] + '.h5'), 'w') as f:
            f.create_dataset('data', data=y_data) 

#convert_npy_2_hdf5(path_to_external_data, path_to_external_data_hd5f)

# %%
#reader_image = h5py.File(os.path.join(path_to_external_data_hd5f,'data/', '02014629_KO_MCA_x.npy'), 'r')
#x = reader_image['data'][()]

#%%
#

def write_file_dict(path, save=True, overwrite=False):
    file_attr = []
    save_path = os.path.join(path,'file_dict.csv')
    file_df = None
    if not os.path.exists(save_path) or overwrite:
        functions = {'mean': np.mean,
                    'max': np.max,
                    'min': np.min,
                    'var': np.var,
                    '95per': lambda x: np.percentile(x, 95),
                    '99per': lambda x: np.percentile(x, 99),
                    '5per': lambda x: np.percentile(x, 5),
                    '1per': lambda x: np.percentile(x, 1),
                    '10per': lambda x: np.percentile(x, 10),
                    '90per': lambda x: np.percentile(x, 90),
                    'median': np.median}

        for file_name in os.listdir(path):
            print(file_name)
            if file_name.endswith('_x.h5'):
                file_path = os.path.join(path, file_name)
                reader_image = h5py.File(file_path, 'r')
                data = reader_image['data'][()]


                file_dict = dict()
                file_dict['name'] = file_name[:-3]
                for func_name in functions.keys():
                    file_dict[func_name] = functions[func_name](data)
                file_attr.append(file_dict)

            elif file_name.endswith('_x.npy'):
                file_path = os.path.join(path, file_name)
                data = np.load(file_path, 'r')

                file_dict = dict()
                file_dict['name'] = file_name[:-4]
                for func_name in functions.keys():
                    file_dict[func_name] = functions[func_name](data)

                file_attr.append(file_dict)
        file_df = pd.DataFrame(file_attr)
        if save:
            file_df.to_csv(os.path.join(path,'file_dict.csv'))

    return file_df

write_file_dict("/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias666/data", overwrite=True)
# %%
#for fold in os.listdir(os.path.join(path_to_external_data_hd5f, 'folds')):
#    fold_path = os.path.join(path_to_external_data_hd5f, 'folds', fold)
#    for split_set in os.listdir(fold_path):
#        write_file_dict(os.path.join(fold_path, split_set))
#path = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias111/data"
#for el in os.listdir(path):
#    os.rename(os.path.join(path, el), os.path.join(path, el.replace("__" ,"_")))
# %%
#df = pd.read_csv('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_hdf5/folds/0/train/file_dict.csv')
#x,y = read_data_names(path_to_external_data)

def integrate_intensity(img, save_path=None):
    inten_x = np.sum(img, axis=(1,2))
    inten_y = np.sum(img, axis=(0,2))
    inten_z = np.sum(img, axis=(0,1))
    plt.figure()
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
"""
for x_file in x:
    img = np.load(os.path.join(path_to_external_data, x_file))
    save_path = os.path.join('/scratch/lhauptmann/BrainSeg/data_analysis/data_intensities', x_file[:-4] + '.png')
    integrate_intensity(img, save_path)
"""

def copy_labels_and_images(path, label_dir, image_dir, name_base, overwrite=False):

    i = 0

    file_dirs = list(os.listdir(path))
    file_dirs = sorted(file_dirs)
    for file_dir in file_dirs:
        
        file_path = os.path.join(path, file_dir)
        if not os.path.isdir(file_path):
            continue
        
        all_elements = list(os.listdir(file_path))
        save_name_image = name_base + '_' + "{:03d}".format(i) + '_0000.nii.gz'
        save_name_label = name_base + '_' + "{:03d}".format(i) + '.nii.gz'
        ############ LOOK FOR IMAGE ###############
        tofs = [el for el in all_elements if 'tof' in el.lower() and (el.endswith('.nii.gz') or el.endswith('.nii'))]
        if len(tofs) == 1:
            image_name = tofs[0]
        else:
            print('Could not find image. Select manually...')
            print(all_elements)
            image_name =  input()

        image_source_path = os.path.join(file_path, image_name)
        image_save_path = os.path.join(image_dir, save_name_image)
        print(image_name, save_name_image)
        # Get affine of image
        img = nib.load(image_source_path)
        affine_new = img.affine.copy()

        nib.save(nib.Nifti1Image(img.dataobj, affine_new, img.header), image_save_path)
        #if not os.path.exists(image_save_path) or overwrite:
        #    shutil.copy(image_source_path, image_save_path)

        ############## LOOK FOR TARGET ####################
        segs = [el for el in all_elements if 'segmentation' in el.lower() and (el.endswith('.nii.gz') or el.endswith('.nii'))]
        if len(segs) == 1:
            image_name = segs[0]
        else:
            print('Could not find target. Select manually...')
            print(all_elements)
            image_name =  input()

        mapping_path = os.path.join(file_path, 'label_assignment.csv')
        label_source_path = os.path.join(file_path, image_name)
        label_save_path = os.path.join(label_dir, save_name_label)
        print(image_name, save_name_label)
        if not os.path.exists(label_save_path) or overwrite:
            transform_labels_nifti(label_source_path, label_save_path, mapping_path, affine_new, binarize_target = 4)

        i = i+1
        

        
def transform_labels_nifti(source, target, mapping_path, affine, binarize_target = -1):
    nii = nib.load(source)
    label_mapping = pd.read_csv(mapping_path, dtype={'class_id': int, 'id_in_file': float, 'class_name': str, 'name_in_table': str})

    data_corr = correct_labels(nii.get_fdata(), label_mapping)
    if binarize_target != -1:
        data_corr = np.where(data_corr == 4, 1,0)
    nii_corr = nib.Nifti1Image(data_corr.astype(int), affine, nii.header, dtype=nii.get_data_dtype())
    nib.save(nii_corr, target)

path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_new'
label_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/nnUNet_raw_data_base/nnUNet_raw_data/Task544_BrainArtery/labelsTr'
image_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/nnUNet_raw_data_base/nnUNet_raw_data/Task544_BrainArtery/imagesTr'
name_base = 'BrainArtery'
copy_labels_and_images(path, label_dir, image_dir, name_base, overwrite=True)


