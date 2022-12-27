#%%
import nibabel as nib
import numpy as np
import os
import nilearn
import nipype.interfaces.fsl as fsl
from nilearn.plotting import plot_anat, plot_roi
from nilearn.masking import compute_brain_mask
from scipy.ndimage import binary_fill_holes, binary_closing
import matplotlib.pyplot as plt
import scipy


#%%
def skull_stripping(in_path, out_dir):
    print(f'\nProcessing {in_path}...')
    btr = fsl.BET()
    btr.inputs.in_file = in_path
    btr.inputs.frac = 0.4
    btr.inputs.out_file = os.path.join(out_dir, os.path.basename(in_path))
    btr.robust = True
    btr.cmdline

    res = btr.run() 

    return res

def extract_brain(img, add_border = 40):
    mask = compute_brain_mask(img)
    mask_data = mask.get_fdata()
    (x_s, y_s, z_s) = mask_data.shape
    extended_mask_data = np.zeros((x_s+2*add_border, y_s+2*add_border, z_s+2*add_border))
    extended_mask_data[add_border:-add_border, add_border:-add_border, add_border:-add_border] = mask_data
    closed_data_de = binary_closing(extended_mask_data, iterations=add_border)[add_border:-add_border, add_border:-add_border, add_border:-add_border]
    return nib.Nifti2Image(closed_data_de,mask.affine, mask.header)

def extended_closing(data, iterations):
    (x_s, y_s, z_s) = data.shape
    data_new = np.zeros((x_s + 2*iterations, y_s + 2*iterations, z_s + 2*iterations))
    data_new[iterations:-iterations,iterations:-iterations,iterations:-iterations] = data
    data_new = scipy.ndimage.binary_closing(data_new, iterations=iterations)
    data = data_new[iterations:-iterations,iterations:-iterations,iterations:-iterations]
    return data

def threshold_masking(data):


    mask = scipy.ndimage.binary_opening(data > np.percentile(data, 99)/10, iterations=2)
    mask = extended_closing(mask, iterations=30)
    return mask


#%%

mri_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
out_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data'
mask_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data'
# %%

#skull_stripping('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/10576901_KJ_ICA_x.nii.gz', out_dir)

#%%
save_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_tm_nifti/data'
files = sorted(list(os.listdir(mri_path)))
files = [os.path.join(mri_path, el) for el in files]
x_files = sorted([f for f in files if f.endswith('_x.nii.gz')])
y_files = sorted([f for f in files if f.endswith('_y.nii.gz')])

lost_points_all = []
lost_points_target_all = []
# %%
for x, y in zip(x_files,y_files):

    label = nib.load(y)
    image = nib.load(x)
    #mask = nib.load(os.path.join(mask_path, os.path.basename(x).replace('_x', '_mask') ))
    #mask_data = mask.get_fdata()
    #mask = extract_brain(image)
    #nib.save(mask, os.path.join(save_path, os.path.basename(y).replace('_y', '_mask')))
    #label_data, mask_data = label.get_fdata(), mask.get_fdata()
    label_data = label.get_fdata()
    mask_data = threshold_masking(image.get_fdata())
    lost_points = np.argwhere(np.logical_and(mask_data < 0.5, label_data!=0))
    lost_points_target = np.argwhere(np.logical_and(mask_data > 0.5, label_data==4))
    lost_points_all.append(lost_points)
    lost_points_target_all.append(lost_points_target)
    print(x)
    print(len(lost_points), len(lost_points_target))
    mask = nib.Nifti1Image(mask_data, image.affine, image.header, dtype='int16')
    save_path_file = os.path.join(save_path, os.path.basename(x).replace('_x', '_mask'))
    print(f'Saving to {save_path_file}')
    if not os.path.exists(save_path_file):
        
        nib.save(mask, save_path_file)


print([len(el) for el in lost_points_all])
print([len(el) for el in lost_points_target_all])
# %%


name = '02014629_KO_MCA' 
x = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/{name}_x.nii.gz'
y = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/{name}_y.nii.gz'
mask = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_sk02_nifti/data/{name}_x.nii.gz_mask.nii.gz'


# %%
index = 0
mask_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_be_nifti/data'
x = x_files[index]
y = y_files[index]
#mask = os.path.join(mask_path, os.path.basename(x).replace('_x', '_mask'))
ext = Extractor()
plot_roi(mask, x)
# %%
plot_roi(mask, x)
# %%
x = nib.load(x)
x_data = x.get_fdata()

# %%
ext = Extractor()
prob = ext.run(x_data)
# %%
import h5py
import matplotlib.pyplot as plt
def read_h5(path):
    try:
        reader = h5py.File(path)
        data = reader['data'][()]
    except:
        print(path)
        return None
    return data

def read_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data


path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/ADAM/data'
img_path = '/scratch_net/biwidl311/lhauptmann/segmentation_3D/data_analysis/ADAM'
mask_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_tm_nifti/data'

def plot_images(path, img_path, include_mask=False):
    files = [os.path.join(path,el) for el in list(os.listdir(path))]
    files = sorted(files)
    x_files = [f for f in files if f.endswith('_x.h5')]
    y_files = [f for f in files if f.endswith('_y.h5')]
    mask_files = [f for f in files if f.endswith('_mask.h5')]
    if len(mask_files) == len(x_files):
        include_mask = False
    if include_mask:
        file_list = [{'x_file': x_file, 'y_file': y_file, 'mask_file': mask_file} for x_file, y_file, mask_file in zip(x_files, y_files, mask_files)]
    else:
        file_list = [{'x_file': x_file, 'y_file': y_file} for x_file, y_file in zip(x_files, y_files)]
    #mask_files = [os.path.join(mask_path, el) for el in sorted(list(os.listdir(mask_path))) if el.endswith('_mask.nii.gz')]
    for file in file_list:
        x_file, y_file = file['x_file'], file['y_file']
        if x_file.endswith('.h5'):
            reader = read_h5
        elif x_file.endswith('.nii.gz'):
            reader = lambda x: read_nifti(x).get_fdata()
        else:
            print('Unknown file type')
            return
        if include_mask:
            mask_file = file['mask_file']
            mask = reader(mask_file)
        x, y = reader(x_file), reader(y_file)
        
    
        if include_mask:
            if not (x.shape == y.shape == mask.shape):
                print(os.path.basename(x_file), os.path.basename(y_file))

                print(x.shape, y.shape, mask.shape)
        elif not (x.shape == y.shape):
            print(x_file)
            print(x.shape, y.shape)
        index = x.shape[2]//2
        plt.imshow(x[:,:,index])
        plt.imshow(y[:,:,index], alpha=0.5)
        if include_mask:
            plt.imshow(mask[:,:,index], alpha=0.5)
        
        save_path_file = os.path.join(img_path, os.path.basename(x_file))[:-5] + '.png'
        print(f'Saving to {save_path_file}\n')
        plt.savefig(save_path_file)
        plt.clf()
        
#plot_images(path, img_path, include_mask=False)

# %%
