#%%
import numpy as np # data handling
import os # path handling
import nibabel as nib # mri nii file reading 
import pandas as pd # for general table handling
import h5py
import multiprocessing
import pandas as pd

# %%

# Write all important paths for read/save here
# if you don't know them, just ommit them (might evoke some errors)
path_to_repo = "/scratch_net/biwidl311/lhauptmann/segmentation_3D"
path_to_data = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/ADAM_Challenge/ADAM_release_subjs/"
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/ADAM/data'
path_to_exterinal_header = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/ADAM/header'

import sys
sys.path.append(os.path.join(path_to_repo, "src/utils/"))


# %%
# Iterate through all folders, and save names of files
MRI_file_list = []
pre = "pre"

for mri_file_name in os.listdir(path_to_data):
    mri_file_path = os.path.join(path_to_data, mri_file_name)
    if os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
        
        MRI_file = {'name': mri_file_name,
            'path': mri_file_path,
            'nii': [],
           'nrrd': [],
           'vtk': [],
           'ctbl': []}

        for mri_file_content_name in list(os.listdir(mri_file_path)):
            mri_file_content_path = os.path.join(mri_file_path, mri_file_content_name)
            if mri_file_content_name.endswith(".nii") or mri_file_content_name.endswith(".nii.gz"):
                MRI_file['nii'].append(mri_file_content_path)

        pre_path = os.path.join(mri_file_path,pre)
        for mri_file_content_name in list(os.listdir(pre_path)):
            mri_file_content_path = os.path.join(pre_path, mri_file_content_name)
            if mri_file_content_name.endswith(".nii") or mri_file_content_name.endswith(".nii.gz"):
                MRI_file['nii'].append(mri_file_content_path)

        MRI_file_list.append(MRI_file)

# sort the list alphabetically
MRI_file_list = np.array(MRI_file_list)
MRI_file_list = MRI_file_list[np.argsort([el['name'] for el in MRI_file_list])]             
        

# %%
npy=False
out_shape= (560,640, 200)
voxel_size = (0.3, 0.3, 0.6)


# %%
# Read through all MRI files, correct the labels and save them as extra files
from nibabel.processing import resample_to_output, conform
from preprocessing_utils import correct_labels, correct_bias, map_labels
# %%
def compute_new_dim(old_dims, old_vox, new_vox):
    assert len(old_dims)==len(old_vox) == len(new_vox)
    new_dim = [0]*len(old_dims)
    for i, (od, ov, nv) in enumerate(zip(old_dims, old_vox, new_vox)):
        new_dim[i] = int(od * (ov/nv))
    return new_dim

def process_file(mri_file, i):
    
    print(f"Processing {i} out of {len(MRI_file_list)}: {mri_file['name']}")

    path_to_saved_y_file = os.path.join(path_to_external_data, mri_file['name'] + '_y')
    path_to_saved_x_file = os.path.join(path_to_external_data, mri_file['name'] + '_x')

    # Process Angiography TOF file (X)
    tof = []
    for nii_file in mri_file['nii'] :
        if 'tof' in nii_file.lower():
            tof.append(nii_file)
    if len(tof) != 1:
        print('Error: Could not find the correct nrrd TOF file')
        print('Files found: ', tof)

    nii_x_img = nib.load(tof[0])

    x_affine = nii_x_img.affine.copy()

    #nii_x_data_bias, _ = correct_bias(nii_x_img.get_fdata())
    nii_x_data_bias = nii_x_img.get_fdata()

    nii_x_img = nib.Nifti1Image(nii_x_data_bias, nii_x_img.affine, nii_x_img.header)

    # preprocessing data
    new_x_dim = compute_new_dim(nii_x_img.header["dim"][1:4], nii_x_img.header["pixdim"][1:4], voxel_size)
    nii_x_img = conform(nii_x_img, voxel_size = voxel_size, out_shape = new_x_dim, order = 3, cval=0)
    #nii_x_img = conform(nii_x_img, out_shape=out_shape, voxel_size=voxel_size, order=3, cval=0, orientation='RAS')
    json_object = pd.Series(nii_x_img.header).to_json()
    with open(os.path.join(path_to_exterinal_header, mri_file['name'] + '.json'), 'w') as f:
        f.write(json_object)

    if npy:
        with open(path_to_saved_x_file + ".npy", 'wb') as f:
            np.save(f, nii_x_img.get_fdata())
    else:
        with h5py.File(path_to_saved_x_file + ".h5", 'w') as f:
            f.create_dataset('data', data=nii_x_img.get_fdata()) 

    # Process Segmentation label file (Y)
    seg = []
    for nii_file in mri_file['nii']:
        if nii_file not in tof and "aneurysms" in nii_file:
            seg.append(nii_file)
    if len(seg) != 1:
        print('Warning: Found multiple segmentation files')
        print('Files found: ', seg)
        
    nii_y_img = nib.load(seg[0])
    nii_y_img = nib.Nifti1Image(nii_y_img.dataobj, x_affine, nii_x_img.header)



    new_y_dim = compute_new_dim(nii_y_img.header["dim"][1:4], nii_y_img.header["pixdim"][1:4], voxel_size)
    nii_y_img = conform(nii_y_img, voxel_size = voxel_size, out_shape = new_y_dim, order = 0, cval=0)

    #nii_y_img = resample_to_output(nii_y_img, voxel_sizes=voxel_size, order = 0, mode = 'constant', cval=0)
    #nii_y_img = conform(nii_y_img, out_shape=out_shape, voxel_size=voxel_size, order=0, cval=0, orientation='RAS')

    if any(nii_y_img.header['dim'] != nii_x_img.header['dim']):
        print('Warning: image and segmentation do not have the same dimensions. ', nii_y_img.header['dim'], nii_x_img.header['dim'])

    if any(nii_y_img.header['pixdim'] != nii_x_img.header['pixdim']):
        print('Warning: image and segmentation do not have the same voxel size. ', nii_y_img.header['pixdim'], nii_x_img.header['pixdim'])


    #label_mapping = pd.read_csv(os.path.join(mri_file['path'], 'label_assignment.csv'), dtype={'class_id': int, 'id_in_file': float, 'class_name': str, 'name_in_table': str})
    # print mapping
    #label_mapping[['id_in_file', 'class_id']].dropna(axis='index', how='any').set_index('id_in_file').to_dict()['class_id']

    nii_y_data = nii_y_img.get_fdata().astype('uint8')
    label_mapping = {1:4}
    nii_y_data_corr = map_labels(nii_y_data, label_mapping)

    if npy:
        with open(path_to_saved_y_file + ".npy", 'wb') as f:
            np.save(f, nii_y_data_corr)
    else:
        with h5py.File(path_to_saved_y_file + ".h5", 'w') as f:
            f.create_dataset('data', data=nii_y_data_corr) 

def run_process(every_n = 4, start_i = 0):
    for i in range(start_i, len(MRI_file_list), every_n):
        process_file(MRI_file_list[i], i)
ps = []
n = 8
for k in range(n):
    ps.append(multiprocessing.Process(target=run_process, args = (n,k,)))

for k in range(n):
    ps[k].start()

for k in range(n):
    ps[k].join()



