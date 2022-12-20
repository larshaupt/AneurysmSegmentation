#%%
import numpy as np # data handling
import os # path handling
import nibabel as nib # mri nii file reading 
import pandas as pd # for general table handling
import h5py
import multiprocessing

# %%

# Write all important paths for read/save here
# if you don't know them, just ommit them (might evoke some errors)
path_to_repo = "/scratch/lhauptmann/BrainSeg/"
path_to_data = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_new"
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_processed/data'
path_to_exterinal_header = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_processed/header'
path_to_preprocessed_nifti = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
path_to_masks = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data'
import sys
sys.path.append(os.path.join(path_to_repo, "src/utils/"))


# %%
# Iterate through all folders, and save names of files
def extract_files(path_to_data):
    MRI_file_list = []

    for mri_file_name in os.listdir(path_to_data):
        mri_file_path = os.path.join(path_to_data, mri_file_name)
        if os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
            
            MRI_file = {'name': mri_file_name,
                'path': mri_file_path,
                'nii': [],
            'nrrd': [],
            'vtk': [],
            'ctbl': []}

            for mri_file_content_name in os.listdir(mri_file_path):
                mri_file_content_path = os.path.join(mri_file_path, mri_file_content_name)
                if mri_file_content_name.endswith(".nii") or mri_file_content_name.endswith(".nii.gz"):
                    MRI_file['nii'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".nrrd"):
                    MRI_file['nrrd'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".vtk"):
                    MRI_file['vtk'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".ctbl"):
                    MRI_file['ctbl'].append(mri_file_content_path)
            MRI_file_list.append(MRI_file)

    # sort the list alphabetically
    MRI_file_list = np.array(MRI_file_list)
    MRI_file_list = MRI_file_list[np.argsort([el['name'] for el in MRI_file_list])]      
    return MRI_file_list

def extract_preprocessed_files(path_to_data, path_to_masks):
    MRI_file_list = []
    masks = []
    x_files = []
    y_files = []
    

    for mri_file_name in os.listdir(path_to_data):
        
        mri_file_path = os.path.join(path_to_data, mri_file_name)
        if not os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):

            if mri_file_path.endswith('_x.nii.gz'):
                x_files.append(mri_file_path)
            elif mri_file_path.endswith('_y.nii.gz'):
                y_files.append(mri_file_path)
    
    for mri_file_name in os.listdir(path_to_masks):
        mri_file_path = os.path.join(path_to_masks, mri_file_name)
        if not os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
            if mri_file_path.endswith('_mask.nii.gz'):

                masks.append(mri_file_path)
    assert len(masks) == len(x_files) == len(y_files)
    masks, x_files, y_files = sorted(masks), sorted(x_files), sorted(y_files)
    
    for x,y,m in zip(x_files, y_files, masks):
        MRI_file_list.append({'x': x, 'y':y, 'mask':m, 'name': os.path.basename(x)[:-8]})

    return MRI_file_list

MRI_file_list = extract_files(path_to_data)

        


# %%
# Read through all MRI files, correct the labels and save them as extra files
from nibabel.processing import resample_to_output, conform
from preprocessing_utils import correct_labels, correct_bias
# %%

def compute_new_dim(old_dims, old_vox, new_vox):
    assert len(old_dims)==len(old_vox) == len(new_vox)
    new_dim = [0]*len(old_dims)
    for i, (od, ov, nv) in enumerate(zip(old_dims, old_vox, new_vox)):
        new_dim[i] = int(od * (ov/nv))
    return new_dim

def do_bias_correction(img):
    img_data_corr = correct_bias(img.get_fdata())[0]
    return nib.Nifti1Image(img_data_corr, img.affine, img.header)


def process_file(mri_file, i, path, resample = True, voxel_size = (0.3, 0.3, 0.6) ,preprocessed = False,bias_corr = False, include_mask=False ,save_header = True, save_as="h5", overwrite=False, label_mapping='label_assignment.csv', skip_tof=False):
    
    print(f"Processing {i} out of {len(MRI_file_list)}: {mri_file['name']}")

    path_to_saved_y_file = os.path.join(path, mri_file['name'] + '_y')
    path_to_saved_x_file = os.path.join(path, mri_file['name'] + '_x')
    path_to_saved_mask = os.path.join(path, mri_file['name'] + '_mask')

    if not skip_tof:
        if not os.path.exists(path_to_saved_x_file) or overwrite:
            # Process Angiography TOF file (X)
            if not preprocessed:
                tof = []
                for nii_file in mri_file['nii'] :
                    if 'tof' in nii_file.lower():
                        tof.append(nii_file)
                if len(tof) != 1:
                    print('Error: Could not find the correct nii TOF file')
                    print('Files found: ', tof)
                nii_x = tof[0]

            else:
                nii_x = mri_file['x'] 

            nii_x_img = nib.load(nii_x)

            x_affine = nii_x_img.affine.copy()
            if bias_corr:
                nii_x_img = do_bias_correction(nii_x_img)


            # preprocessing data
            #nii_x_img = resample_to_output(nii_x_img, voxel_sizes=voxel_size, order = 3, mode = 'constant', cval=0)
            if resample:
                new_x_dim = compute_new_dim(nii_x_img.header["dim"][1:4], nii_x_img.header["pixdim"][1:4], voxel_size)
                nii_x_img = conform(nii_x_img, voxel_size = voxel_size, out_shape = new_x_dim, order = 3, cval=0)

            if save_header:
                json_object = pd.Series(nii_x_img.header).to_json()
                with open(os.path.join(path_to_exterinal_header, mri_file['name'] + '.json'), 'w') as f:
                    f.write(json_object)

            if save_as == "npy":
                with open(path_to_saved_x_file + ".npy", 'wb') as f:
                    np.save(f, nii_x_img.get_fdata())
            elif save_as == "h5":
                with h5py.File(path_to_saved_x_file + ".h5", 'w') as f:
                    f.create_dataset('data', data=nii_x_img.get_fdata()) 
            elif save_as == "nifti":
                nib.save(nii_x_img, path_to_saved_x_file + ".nii.gz")

    if not os.path.exists(path_to_saved_y_file) or overwrite:
        # Process Segmentation label file (Y)
        if not preprocessed:
            seg = []
            for nii_file in mri_file['nii']:
                if nii_file not in tof:
                    seg.append(nii_file)
            if len(seg) != 1:
                print('Warning: Found multiple segmentation files')
                print('Files found: ', seg)
                cands = [el for el in seg if "segmentation" in el.lower()]
                if cands != None:
                    seg = cands
            nii_y = seg[0]
        else:
            nii_y = mri_file['y']
            
        nii_y_img = nib.load(nii_y)
        # Copy affine from X
        nii_y_img = nib.Nifti1Image(nii_y_img.dataobj, x_affine, nii_x_img.header)

        if resample:
            new_y_dim = compute_new_dim(nii_y_img.header["dim"][1:4], nii_y_img.header["pixdim"][1:4], voxel_size)
            nii_y_img = conform(nii_y_img, voxel_size = voxel_size, out_shape = new_y_dim, order = 0, cval=0)
        #nii_y_img = resample_to_output(nii_y_img, voxel_sizes=voxel_size, order = 0, mode = 'constant', cval=0)
        #nii_y_img = conform(nii_y_img, out_shape=out_shape, voxel_size=voxel_size, order=0, cval=0, orientation='RAS')

        if any(nii_y_img.header['dim'] != nii_x_img.header['dim']):
            print('Warning: image and segmentation do not have the same dimensions. ', nii_y_img.header['dim'], nii_x_img.header['dim'])

        if any(nii_y_img.header['pixdim'] != nii_x_img.header['pixdim']):
            print('Warning: image and segmentation do not have the same voxel size. ', nii_y_img.header['pixdim'], nii_x_img.header['pixdim'])

        if label_mapping != None and label_mapping != "":
            if isinstance(label_mapping, dict):
                label_mapping_dict ==label_mapping
            elif isinstance(label_mapping, str):
                label_mapping_dict = pd.read_csv(os.path.join(mri_file['path'], label_mapping), dtype={'class_id': int, 'id_in_file': float, 'class_name': str, 'name_in_table': str})
            else:
                raise Exception(f"{label_mapping} does not have the right format")

            nii_y_data_corr = correct_labels(nii_y_img.get_fdata().astype('uint8'), label_mapping_dict)
            nii_y_img = nib.Nifti1Image(nii_y_data_corr, nii_y_img.affine, nii_y_img.header)

        if save_as == "npy":
            with open(path_to_saved_y_file + ".npy", 'wb') as f:
                np.save(f, nii_y_img.get_fdata())
        elif save_as == "h5":
            with h5py.File(path_to_saved_y_file + ".h5", 'w') as f:
                f.create_dataset('data', data=nii_y_img.get_fdata()) 
        elif save_as == "nifti":
            nib.save(nii_y_img,path_to_saved_y_file + ".nii.gz" )

    if include_mask:
        if preprocessed:
            nii_mask = mri_file['mask']

        nii_mask_img = nib.load(nii_mask)
        # Copy affine from X
        nii_mask_img = nib.Nifti1Image(nii_mask_img.dataobj, x_affine, nii_x_img.header)

        if resample:
            new_mask_dim = compute_new_dim(nii_mask_img.header["dim"][1:4], nii_mask_img.header["pixdim"][1:4], voxel_size)
            nii_mask_img = conform(nii_mask_img, voxel_size = voxel_size, out_shape = new_mask_dim, order = 0, cval=0)


        if save_as == "npy":
            with open(path_to_saved_mask + ".npy", 'wb') as f:
                np.save(f, nii_mask_img.get_fdata())
        elif save_as == "h5":
            with h5py.File(path_to_saved_mask + ".h5", 'w') as f:
                f.create_dataset('data', data=nii_mask_img.get_fdata()) 
        elif save_as == "nifti":
            nib.save(nii_mask_img,path_to_saved_mask + ".nii.gz" )


MRI_file_list = extract_files(path_to_data)
#MRI_file_list = extract_preprocessed_files(path_to_preprocessed_nifti, path_to_masks)

save_path = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data"

targets = ["11096773_IB_PComm", "10919238_GD_MCA"]
label_mapping_file = "label_assignment.csv"
def run_process(every_n = 4, start_i = 0):
    for i in range(start_i, len(MRI_file_list), every_n):
        if targets != None:
            if not any([target in MRI_file_list[i]["name"] for target in targets]):
                continue
        process_file(MRI_file_list[i], 
        i, 
        save_path, 
        resample = False, 
        voxel_size = (1.0, 1.0, 1.0) ,
        bias_corr = True, 
        preprocessed=False,
        save_header = False, 
        save_as="nifti", 
        overwrite=True, 
        label_mapping=label_mapping_file,
        include_mask=False,
        )

ps = []
n = 8
split_dif = 8
split_id = 0
for k in range(split_id*split_dif, split_dif*(split_id+1)):
    ps.append(multiprocessing.Process(target=run_process, args = (n,k,)))

for k in range(len(ps)):
    ps[k].start()

for k in range(len(ps)):
    ps[k].join()



# %%

