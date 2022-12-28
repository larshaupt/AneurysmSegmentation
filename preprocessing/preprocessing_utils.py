#%%
import numpy as np # data handling
import os # path handling
import matplotlib.pyplot as plt # visualization
import nrrd # mri nrrd file reading
import nibabel as nib # mri nii file reading
import time 
import vtk # to handel mri files
import seaborn as sns # for beautiful plots
import json # to save header files
import openpyxl # for reading for excel tables
import SimpleITK as sitk
import torch
import h5py
#%%
# saves the given mri data as gif


################ Save Volumes ##########################
def animate_img(img_data, size='small', path_to_data_analysis = "../data_analysis/", save_path = None):

    import matplotlib.animation as animate
    imgs = []
    fig = plt.figure()
    if save_path == None:
        save_path = os.path.join(path_to_data_analysis, 'gifs/', str(int(time.time())) + '_gif.gif')

    if size == 'small':
        step = 4
    else:
        step = 1

    for i in np.arange(0,img_data.shape[2], step):
        im = plt.imshow(img_data[::step,::step,i], animated=True)
        imgs.append([im])

    ani = animate.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat_delay=500)
    plt.axis('off')
    ani.save(save_path)
    plt.show()
    print('Gif saved under: ', save_path)

def save_to_nifti(data, path, voxel_size):
    nifti = nib.Nifti1Image((data).astype(np.int16), np.diag([*voxel_size, 0]))

    #nifti.header['pixdim'] = [1.,voxel_size[0],voxel_size[1],voxel_size[2],0.,0.,0.,0.]
    #nifti.header['srow_x'] =  [voxel_size[0],0.,0.,0.]
    #nifti.header['srow_y'] =  [0.,voxel_size[1],0.,0.]
    #nifti.header['srow_z'] =  [0.,0.,voxel_size[2],0.]
    nib.save(nifti, path) 

def save_to_h5(data, path):
    if not path.endswith(".h5"):
        path = path + ".h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data) 



######################## Read Volumes #########################

# read in one mri nifti file and print file and header
def read_in_nifti(MRI_file, plot=False):
    print(MRI_file['nii'][0])

    nii_img = nib.load(MRI_file['nii'][0])
    nii_data =  nii_img.get_fdata()
    if plot:
        plt.imshow(nii_data[:,:,80])
    print(nii_img.header)
    return nii_data


def read_h5(path):
    reader = h5py.File(path)
    data = reader['data'][()]
    return data

# reads in one nii file and prints image and header
def read_in_nrrd(MRI_file, plot=False):
    nrrd_x_data, nrrd_x_header = nrrd.read(MRI_file['nrrd'][0])
    print(MRI_file['nrrd'][0])
    print(nrrd_x_header)
    if plot:
        plt.imshow(nrrd_x_data[:,:,80])
    return nrrd_x_data


# reads in volume data
def read_data_names(path_to_processed_data, check_dim= False, full_path=False, keep_ending=True):

    # reads in all npy files

    x = []
    y = []

    # puts all the filenames in x and y
    for file_name in os.listdir(path_to_processed_data):
        file_path = os.path.join(path_to_processed_data, file_name)
        if file_path.endswith('_x.npy') or file_path.endswith('_x.h5'):
            x.append(file_name)
        elif file_path.endswith('_y.npy') or file_path.endswith('_y.h5'):
            y.append(file_name)
        else: 
            print('Error: ', file_name)

    # put corresponding elements at the same index into the list        
    x.sort()
    y.sort()

    if full_path:    
        x = [os.path.join(path_to_processed_data, el) for el in x]
        y = [os.path.join(path_to_processed_data, el) for el in y]

    if not keep_ending: 
        x = [el[:-el[::-1].find('.')-1] for el in x]
        y = [el[:-el[::-1].find('.')-1] for el in y]

    
    # to check if dimensions fit correctly, can be ommited
    if check_dim:
        for i in range(len(x)):
            x_data = np.load(os.path.join(path_to_processed_data,x[i]))
            y_data = np.load(os.path.join(path_to_processed_data,y[i]))
            if x_data.shape != y_data.shape:
                print(i, ': ',x[i], y[i],'Dim not fitting')
                print(x_data.shape, y_data.shape)
            
    x = np.array(x)
    y = np.array(y)

    return x,y


# reads in headers and compute voxel size

def read_header_data(path_to_headers):

    x_headers = []
    y_headers = []
    voxel_volumes = []

    for header_file in sorted(os.listdir(path_to_headers)):
        #print(header_file)
        if header_file.endswith('_x.json'):
            with open(os.path.join(path_to_headers, header_file)) as f:
                x_headers.append(json.load(f))

        elif header_file.endswith('_y.json'):
            with open(os.path.join(path_to_headers, header_file)) as f:
                y_headers.append(json.load(f))

    for i, x_header in enumerate(x_headers):
        #if x_header['voxel spacing'] != y_headers[i]['voxel spacing']:
        #    print(x_header['voxel spacing'],y_headers[i]['voxel spacing'] )

        # --> Voxel sizes are the same (up to rounding errors) between segmentation and anatomy file
        if 'voxel spacing' in x_header.keys():
            voxel_volume = x_header['voxel spacing']
        else:
            voxel_volume = x_header['pixdim'][1:3]
        voxel_volumes.append(voxel_volume)

    return x_headers, y_headers, voxel_volumes



###################### Label Correction ################################

# function for creating the label mappings
from difflib import SequenceMatcher
def match_labels(color_table, class_table):
    mapping = {}
    i = 0
    js = list(range(class_table.shape[0]))
    matches_ratio = []
    
    while len(js) != 0 and i < color_table.shape[0]:
        matches = [SequenceMatcher(None, class_table['name'].iloc[j], color_table['name'].iloc[i]).ratio() for j in js]
        j = js[matches.index(max(matches))]
        mapping[j] = [class_table['label id'].iloc[j],color_table['label id'].iloc[i] ,class_table['name'].iloc[j], color_table['name'].iloc[i]]
        i = i+1
        js.remove(j)
        matches_ratio.append(max(matches))
        
    if np.mean(matches_ratio) < 0.75:
        print('Warning: Mapping did not work correctly', np.mean(matches_ratio))
        
    for j in js:
        mapping[j] = [j, None, class_table['name'][j], None]
        
    return mapping


# takes data and label mappings as input and return corrected data
def correct_labels(data, mapping):
    dtype = data.dtype
    label_mapping = mapping[['id_in_file', 'class_id']].dropna(axis='index', how='any').astype(int).set_index('id_in_file').to_dict()['class_id']
    # bugfix
    return map_labels(data, label_mapping).astype(int)

def map_labels(data, label_mapping:dict):
    l_keys = []
    for key in label_mapping.keys():
        if label_mapping[key] != key:
            l_keys.append(key)

    """with np.nditer(data_new, op_flags=['readwrite']) as it:
        for value in it:"""

    data_new = np.copy(data)
    x_dim, y_dim, z_dim = data_new.shape
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if int(data_new[x,y,z]) in l_keys:
                    #print('Former: ', value)
                    data_new[x,y,z] = int(label_mapping[int(data_new[x,y,z])])
                    #print('New: ', value, '\n')
                    #print('Here! Changing ', value, ' to ', label_mapping[int(value)])
        
    #print(np.sum(data != data_new))
    return data_new
        
    


################## Conversion ##########################

# function for converting nrrd to nifti file
def convert_nrrd_to_nifti(filepath):

    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filepath)
    reader.Update()
    info = reader.GetInformation()
    image = reader.GetOutput()

    new_filepath = filepath[:-5] # remove the '.nrrd'
    new_filepath = new_filepath + '.nii.gz'



    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(new_filepath)
    writer.SetInformation(info)
    writer.Write()


###################### Data Preprocessing ############################

# executes N4 bias field correction on the single image
def correct_bias(img_data):
    img_data_sitk = sitk.GetImageFromArray(img_data)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #corrector.SetMaximumNumberOfIterations([5,5,5,5])
    #mask_image = sitk.OtsuThreshold(img_data, 0, 1, 200)
    corrected_image = corrector.Execute(img_data_sitk)
    log_bias_field = corrector.GetLogBiasFieldAsImage(img_data_sitk)
    
    return sitk.GetArrayFromImage(corrected_image), sitk.GetArrayFromImage(log_bias_field)

######################### Visualization ##########################

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

def plot_overlay(image, label, slice:int, alpha=0.3):
    assert image.shape == label.shape
    fig = plt.figure()
    ax1 = plt.imshow(image[...,slice])
    ax2 = plt.imshow(label[..., slice], alpha=alpha)
    plt.show()
