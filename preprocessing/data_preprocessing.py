#%%
import numpy as np # data handling
import os # path handling
import nibabel as nib # mri nii file reading 
import pandas as pd # for general table handling
import h5py
import multiprocessing
import pickle
from nibabel.processing import resample_to_output, conform


from preprocessing_utils import correct_labels, correct_bias

# %%

# Write all important paths for read/save here
# if you don't know them, just ommit them (might evoke some errors)
path_to_repo = "/scratch/lhauptmann/BrainSeg/"
path_to_data = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_new"
path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_processed/data'
path_to_exterinal_header = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_processed/header'
path_to_preprocessed_nifti = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
path_to_masks = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
path_to_Lausanne = '/scratch_net/biwidl311/lhauptmann/Lausanne_ex'



# %%
# Iterate through all folders, and save names of files

def extract_files(path_to_data, path_to_masks):

    """
    Extract the correct and corresponding data, labels and masks from a directory.

    Parameters
    ----------
    path_to_data : str
        Path to the directory containing the data files. The data files should have the format
        "[name]_[f].[ending]" where name is the name of the data point, f is either 'x' or 'y'
        and ending can be '.nii.gz' or '.h5'.
    path_to_masks : str
        Path to the directory containing the mask files. The mask files should have the format
        "[name]_[f].[ending]" where name is the name of the data point, f is either 'mask' or 'atlas'
        and ending can be '.nii.gz' or '.h5'. If no masks are available, pass an empty string.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the 'x', 'y' and 'mask' file paths corresponding to a single data point.
        The dictionary also contains a 'name' key, which gives the name of the data point.
    """

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
    if path_to_masks != '':
        for mri_file_name in os.listdir(path_to_masks):
            mri_file_path = os.path.join(path_to_masks, mri_file_name)
            if not os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
                if mri_file_path.endswith('_atlasn.nii.gz') or mri_file_path.endswith('_mask.nii.gz'):
                    masks.append(mri_file_path)
    else:
        masks = len(x_files)*['']

                    
    assert len(x_files) == len(y_files) == len(masks), print(len(masks), len(x_files), len(y_files))
    masks, x_files, y_files = sorted(masks), sorted(x_files), sorted(y_files)
    
    for x,y,m in zip(x_files, y_files, masks):
        MRI_file_list.append({'x': x, 'y':y, 'mask':m, 'name': os.path.basename(x)[:-9]})

    return MRI_file_list

def extract_files_USZ(path_to_data, path_to_masks):

    """
    Extract files from USZ dataset and return a list of dictionaries with the file paths and information.

    This function is an extended version of the extract_files function and is specifically designed for the USZ dataset. It extracts all the MRI files from the path_to_data directory and the masks from the path_to_masks directory. The extracted files are sorted alphabetically and the resulting list of dictionaries contains the paths of the MRI files along with the file information.

    Parameters:
    path_to_data (str): The path to the directory containing the MRI files.
    path_to_masks (str): The path to the directory containing the masks.

    Returns:
    MRI_file_list (list): A list of dictionaries where each dictionary represents a file and contains the following keys:
    'x' (str): The path to the TOF MRI file.
    'y' (str): The path to the segmentation MRI file.
    'mask' (str): The path to the mask file.
    'name' (str): The name of the file.
    """

    folder_content_list = []
    masks = []


    # extract all files by ending without extracting their meaning (label, data, mask,...)
    for mri_file_name in os.listdir(path_to_data):
        mri_file_path = os.path.join(path_to_data, mri_file_name)
        if os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
            
            folder_content = {'name': mri_file_name,
                'path': mri_file_path,
                'nii': [],
            'nrrd': [],
            'vtk': [],
            'ctbl': []}

            for mri_file_content_name in os.listdir(mri_file_path):
                mri_file_content_path = os.path.join(mri_file_path, mri_file_content_name)
                if mri_file_content_name.endswith(".nii") or mri_file_content_name.endswith(".nii.gz"):
                    folder_content['nii'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".nrrd"):
                    folder_content['nrrd'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".vtk"):
                    folder_content['vtk'].append(mri_file_content_path)
                elif mri_file_content_name.endswith(".ctbl"):
                    folder_content['ctbl'].append(mri_file_content_path)
            folder_content_list.append(folder_content)

    # sort the list alphabetically
    folder_content_list = np.array(folder_content_list)
    folder_content_list = folder_content_list[np.argsort([el['name'] for el in folder_content_list])]      

    if path_to_masks != '':
        for mri_file_name in os.listdir(path_to_masks):
            mri_file_path = os.path.join(path_to_masks, mri_file_name)
            if not os.path.isdir(mri_file_path) and not mri_file_name.startswith('.'):
                if mri_file_path.endswith('_atlasn.nii.gz') or mri_file_path.endswith('_mask.nii.gz'):
                    masks.append(mri_file_path)

    # extract the files with their meaning (label, data, mask,...)
    for mri_file in folder_content_list:
        tof = []
        for nii_file in mri_file['nii'] :
            print(nii_file.lower())
            if 'tof' in nii_file.lower():
                tof.append(nii_file)
        if len(tof) != 1:
            print(f'Error: Could not find the correct nii TOF file for {mri_file["name"]}')
            print('Files found: ', tof)
            if len(tof) == 0:
                raise ValueError('No TOF file in nii format found. Maybe you need to convert the corresponding tof file')

        nii_x = tof[0]

        seg = []
        for nii_file in mri_file['nii']:
            if nii_file not in tof:
                seg.append(nii_file)
        if len(seg) != 1:
            print('Warning: Found multiple segmentation files')
            print('Files found: ', seg)
            cands = [el for el in seg if ".seg" in el.lower()]
            if cands != None:
                seg = cands
        nii_y = seg[0]

        matching_masks = [m for m in masks if mri_file['name'] if mri_file['name'] in m]
        if len(matching_masks) == 0:
            nii_mask = ''

        MRI_file_list.append({'x': nii_x, 'y':nii_y, 'mask':nii_mask, 'name': mri_file['name']})


    return MRI_file_list

#MRI_file_list = extract_files(path_to_data)


# %%

def compute_new_dim(old_dims, old_vox, new_vox):

    """
    Compute the new dimensions of an object based on its old dimensions and voxel sizes.

    Given the old dimensions old_dims of an object, its current voxel size old_vox and a desired new voxel size new_vox, this function returns the new dimensions of the object that would result from resampling it to the new voxel size.

    The lengths of the old_dims, old_vox and new_vox lists must be equal.

    Args:
    old_dims (list): The old dimensions of the object.
    old_vox (list): The old voxel size of the object.
    new_vox (list): The desired new voxel size of the object.

    Returns:
    list: The new dimensions of the object after resampling to the new voxel size.

    Raises:
    AssertionError: If the lengths of old_dims, old_vox and new_vox are not equal.
    """

    assert len(old_dims)==len(old_vox) == len(new_vox)
    new_dim = [0]*len(old_dims)
    for i, (od, ov, nv) in enumerate(zip(old_dims, old_vox, new_vox)):
        new_dim[i] = int(od * (ov/nv))
    return new_dim

def do_bias_correction(img):
    """
    Apply bias correction to a Nifti1Image.

    Parameters
    ----------
    img : nib.Nifti1Image
        The input Nifti1Image to be corrected.

    Returns
    -------
    nib.Nifti1Image
        The bias-corrected Nifti1Image.

    """

    img_data_corr = correct_bias(img.get_fdata())[0]
    return nib.Nifti1Image(img_data_corr, img.affine, img.header)


def process_file(mri_file,  
        save_path,
        index = 0, 
        resample = True, 
        voxel_size = (0.3, 0.3, 0.6) ,
        bias_corr = False, 
        include_mask=False ,
        save_header = True, 
        save_as="h5", 
        overwrite=False, 
        label_mapping='label_assignment.csv', 
        skip_tof=False, 
        skip_label = False):

    """
    Process a given MRI file using various pre-processing steps.

    The function process_file() processes a given MRI file using various pre-processing steps, including resampling, bias correction, and mask generation. The processed data can be saved as .npy, .h5, or .nii.gz file.

    Input:
    mri_file: A dictionary containing the names of the angiography TOF file (x) and the segmentation label file (y) as keys.
    save_path: The path to the directory where the processed data will be saved.
    index: An optional integer indicating the index of the current file being processed out of all the files in the list.
    resample: An optional boolean flag indicating whether the input data should be resampled. Default is True.
    voxel_size: An optional tuple of three float values representing the voxel size for the resampled data. Default is (0.3, 0.3, 0.6).
    bias_corr: An optional boolean flag indicating whether the input angiography TOF file should undergo bias correction. Default is False.
    include_mask: An optional boolean flag indicating whether a mask should be generated from the input data. Default is False.
    save_header: An optional boolean flag indicating whether the header information of the input data should be saved. Default is True.
    save_as: An optional string indicating the format in which the processed data should be saved. Options are 'npy', 'h5', or 'nifti'. Default is 'h5'.
    overwrite: An optional boolean flag indicating whether the saved data should overwrite any existing data in the save path. Default is False.
    label_mapping: An optional string indicating the path to a CSV file containing the mapping of labels to anatomical structures. Default is 'label_assignment.csv'.
    skip_tof: An optional boolean flag indicating whether the processing of the angiography TOF file should be skipped. Default is False.
    skip_label: An optional boolean flag indicating whether the processing of the segmentation label file should be skipped. Default is False.

    Output:
    None. The processed data and header information (if specified) will be saved to the specified save_path in the specified format.

    """

    print(f"Processing {index} out of {len(MRI_file_list)}: {mri_file['name']}")

    path_to_saved_y_file = os.path.join(save_path, mri_file['name'] + '_y')
    path_to_saved_x_file = os.path.join(save_path, mri_file['name'] + '_x')
    path_to_saved_mask = os.path.join(save_path, mri_file['name'] + '_mask')

    nii_x = mri_file['x'] 

    nii_x_img = nib.load(nii_x)


    x_affine = nii_x_img.affine.copy()
    x_header = nii_x_img.header.copy()
    new_x_dim = compute_new_dim(nii_x_img.header["dim"][1:4], nii_x_img.header["pixdim"][1:4], voxel_size)

    ########### Process Data File (X) ############
    if not skip_tof:
        if not os.path.exists(path_to_saved_x_file) or overwrite:
            # Process Angiography TOF file (X)
        
            if bias_corr:
                nii_x_img = do_bias_correction(nii_x_img)

            if resample:
                nii_x_img = conform(nii_x_img, voxel_size = voxel_size, out_shape = new_x_dim, order = 3, cval=0)

            if save_as == "npy":
                with open(path_to_saved_x_file + ".npy", 'wb') as f:
                    np.save(f, nii_x_img.get_fdata())
            elif save_as == "h5":
                with h5py.File(path_to_saved_x_file + ".h5", 'w') as f:
                    f.create_dataset('data', data=nii_x_img.get_fdata()) 
            elif save_as == "nifti":
                nib.save(nii_x_img, path_to_saved_x_file + ".nii.gz")
    if save_header:
        with open(os.path.join(save_path, mri_file['name'] + '_header.pickle'), 'wb') as f:
            pickle.dump(nii_x_img.header, f, protocol=pickle.HIGHEST_PROTOCOL)

    ########### Process Label File (Y) ############
    if not skip_label:
        if not os.path.exists(path_to_saved_y_file) or overwrite:
            # Process Segmentation label file (Y)
            nii_y = mri_file['y']
                
            nii_y_img = nib.load(nii_y)
            if nii_y_img.get_fdata().ndim != 3:
                print(nii_y_img.get_fdata().ndim)
            # Copy affine from X
            nii_y_img = nib.Nifti1Image(nii_y_img.dataobj, x_affine, x_header)

            if resample:
                nii_y_img = conform(nii_y_img, voxel_size = voxel_size, out_shape = new_x_dim, order = 0, cval=0)
            #nii_y_img = resample_to_output(nii_y_img, voxel_sizes=voxel_size, order = 0, mode = 'constant', cval=0)
            #nii_y_img = conform(nii_y_img, out_shape=out_shape, voxel_size=voxel_size, order=0, cval=0, orientation='RAS')

            if any(nii_y_img.header['dim'] != nii_x_img.header['dim']):
                print('Warning: image and segmentation do not have the same dimensions. ', nii_y_img.header['dim'], nii_x_img.header['dim'])

            if any(nii_y_img.header['pixdim'] != nii_x_img.header['pixdim']):
                print('Warning: image and segmentation do not have the same voxel size. ', nii_y_img.header['pixdim'], nii_x_img.header['pixdim'])

            if label_mapping != None and label_mapping != "":
                if isinstance(label_mapping, dict):
                    label_mapping_dict = label_mapping
                elif isinstance(label_mapping, str):
                    label_mapping_dict = pd.read_csv(os.path.join(mri_file['path'], label_mapping), dtype={'class_id': int, 'id_in_file': float, 'class_name': str, 'name_in_table': str})
                else:
                    raise Exception(f"{label_mapping} does not have the right format")

                nii_y_data_corr = correct_labels(nii_y_img.get_fdata().astype('uint8'), label_mapping_dict)
                nii_y_img = nib.Nifti1Image(nii_y_data_corr, nii_y_img.affine, nii_y_img.header)
            if nii_y_img.get_fdata().shape != nii_x_img.get_fdata().shape:
                print(f"Warning: label and target don't have the same shape: {nii_y_img.get_fdata().shape} and {nii_x_img.get_fdata().shape}")
            if save_as == "npy":
                with open(path_to_saved_y_file + ".npy", 'wb') as f:
                    np.save(f, nii_y_img.get_fdata())
            elif save_as == "h5":
                with h5py.File(path_to_saved_y_file + ".h5", 'w') as f:
                    f.create_dataset('data', data=nii_y_img.get_fdata()) 
            elif save_as == "nifti":
                nib.save(nii_y_img,path_to_saved_y_file + ".nii.gz" )

    ########### Process Mask File ############
    if include_mask and (not os.path.exists(path_to_saved_mask) or overwrite) and os.path.exists(mri_file['mask']):
        nii_mask = mri_file['mask']

        nii_mask_img = nib.load(nii_mask)
        # Copy affine from X
        nii_mask_img = nib.Nifti1Image(nii_mask_img.dataobj, x_affine, x_header)

        if resample:
            #new_mask_dim = compute_new_dim(nii_mask_img.header["dim"][1:4], nii_mask_img.header["pixdim"][1:4], voxel_size)
            nii_mask_img = conform(nii_mask_img, voxel_size = voxel_size, out_shape = new_x_dim, order = 0, cval=0)


        if save_as == "npy":
            with open(path_to_saved_mask + ".npy", 'wb') as f:
                np.save(f, nii_mask_img.get_fdata())
        elif save_as == "h5":
            with h5py.File(path_to_saved_mask + ".h5", 'w') as f:
                f.create_dataset('data', data=nii_mask_img.get_fdata()) 
        elif save_as == "nifti":
            nib.save(nii_mask_img,path_to_saved_mask + ".nii.gz" )


#MRI_file_list = extract_files(path_to_data)
MRI_file_list = extract_files(path_to_preprocessed_nifti, '')
save_path = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias/data"

#targets = ["11096773_IB_PComm", "10919238_GD_MCA"]
targets = []
label_mapping_file = "label_assignment.csv"
def run_process(every_n = 4, start_i = 0):
    for i in range(start_i, len(MRI_file_list), every_n):
        if len(targets) != 0:
            if not any([target in MRI_file_list[i]["name"] for target in targets]):
                continue
        process_file(
        MRI_file_list[i], 
        save_path, 
        index = i,
        resample = True, 
        voxel_size = (0.3, 0.3, 0.6) ,
        bias_corr = False, 
        save_header = False, 
        save_as="h5", 
        overwrite=False, 
        label_mapping=None,
        include_mask=False,
        skip_label=False,
        skip_tof=True,
        )

# Multiprocessing
# If you want to run a single thread, set n = 1
# Otherwise set n to the number of threads you want to use

n = 6
start_from = 0
split_dif = n
split_id = 0


ps = []
for k in range(start_from + split_id*split_dif, start_from + split_dif*(split_id+1)):
    ps.append(multiprocessing.Process(target=run_process, args = (n,k,)))

for k in range(len(ps)):
    ps[k].start()

for k in range(len(ps)):
    ps[k].join()



# %%

