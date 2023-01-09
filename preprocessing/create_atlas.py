# %%
import ants
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import nibabel as nib
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat, plot_roi
data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
transforms_path = '/srv/beegfs02/scratch/brain_artery/data/training/affines'
transforms_path_inv = '/srv/beegfs02/scratch/brain_artery/data/training/affines_inv'
atlas_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/tofAverage.nii.gz'
atlas_vessel_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/vesselProbabilities.nii.gz'
split_dict_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias/k_fold_split2_val.json'



# %%
split_dict = pd.read_json(split_dict_path)
y_test_names = [el[:-2] for el in split_dict[0]['y_test']]

atlas = ants.image_read(atlas_path)
atlas_vessel = ants.image_read(atlas_vessel_path)


all_files = [os.path.join(data_path, f) for f in list(os.listdir(data_path))]
x_files = [f for f in all_files if f.endswith('_x.nii.gz')]
y_files = [f for f in all_files if f.endswith('_y.nii.gz')]
x_files, y_files = sorted(x_files), sorted(y_files)
target_label=4

# %%
target_label = 1 # 4 = aneurysm
data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/ADAM_Challenge/ADAM_release_subjs'
y_files = [os.path.join(data_path, f, 'aneurysms.nii.gz') for f in os.listdir(data_path)]
y_files = sorted(y_files)
# %%

def transform_y(y_file, transforms_path):
    
    y_name = os.path.split(y_file)[1][:-9]

    #y_name = os.path.split(os.path.split(y_file)[0])[1]
    print(y_name)
    transform_fwd_path = os.path.join(transforms_path, y_name + '_fwd.mat')
    transform_inv_path = os.path.join(transforms_path, y_name + '_inv.mat')

    y = ants.image_read(y_file)

    y_transformed = ants.apply_transforms(fixed=atlas, moving=y, transformlist=[transform_inv_path])
    
    return y_transformed
   
y_files_train = [f for f in y_files if not os.path.basename(f)[:-9] in y_test_names]
atlas_aneurysm = atlas.new_image_like(np.zeros(atlas.shape))
i = 0
for y_file in y_files_train:  
    y_transformed = transform_y(y_file, transforms_path)
    # normalization step
    y_transformed = (y_transformed==target_label).astype("float32")
    if y_transformed.sum() != 0:
        #y_transformed = y_transformed/(y_transformed.sum()/1000000)
        print(i)
        atlas_aneurysm = atlas_aneurysm + y_transformed
        i += 1

atlas_aneurysm = atlas_aneurysm / i

# %%

index = 100
plt.imshow(atlas[:, :, index], cmap='gray')

#plt.imshow(atlas_aneurysm_smoothed[:, :, index], alpha=0.5, cmap='jet')

# %%
ants.image_write(atlas_aneurysm, '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/atlasn_aneurysm_adam.nii.gz')
# %%
atlas_aneurysm_usz = ants.image_read('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/atlasn_aneurysm.nii.gz')
atlas_aneurysm_adam = ants.image_read('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/atlasn_aneurysm_adam.nii.gz')
atlas_aneurysm = (atlas_aneurysm_usz + atlas_aneurysm_adam)/2 
#%%
atlas_aneurysm_smoothed = atlas_aneurysm.clone()
atlas_aneurysm_smoothed = atlas_aneurysm_smoothed.apply(lambda x: scipy.ndimage.gaussian_filter(x, sigma=10, truncate=100))

#atlas_aneurysm_smoothed = scipy.ndimage.grey_dilation(atlas_aneurysm_smoothed, size=(10, 10, 10))

# %%
index = 100
plt.imshow(atlas[:, :, index], cmap='gray')
plt.imshow(atlas_aneurysm_smoothed[:, :, index], alpha=0.5, cmap='jet')
#plt.imshow(atlas_vessel[:, :, index], alpha=0.5, cmap='jet')
# %%
ants.image_write(atlas_aneurysm_smoothed, '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/atlasn_aneurysm_smoothed.nii.gz')
# %%
atlas_aneurysm_smoothed = ants.image_read('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas/atlasn_aneurysm_smoothed.nii.gz')
# %%
# Transform back
file_i = 10
y_file = y_files[file_i]
x_file = x_files[file_i]
x, y = ants.image_read(x_file), ants.image_read(y_file)
transform_inv_path = os.path.join(transforms_path_inv, os.path.split(x_file)[1][:-9] + '_inv.mat')
transform_fwd_path = os.path.join(transforms_path_inv, os.path.split(x_file)[1][:-9] + '_fwd.mat')

aneurysms_prior = ants.apply_transforms(fixed=x, moving=atlas_aneurysm_smoothed, transformlist=[transform_fwd_path])

index = 100
plt.imshow(x[:, :, index], cmap='gray')
plt.imshow(aneurysms_prior[:, :, index], alpha=0.5, cmap='jet')
#plt.imshow(atlas_vessel[:, :, index], alpha=0.5, cmap='jet')


# %%

def transform_and_save_atlas(x_file, data_path, transforms_path, atlas_aneurysm_smoothed, overwrite=False):
    print(x_file)
    x_name = os.path.split(x_file)[1][:-9]
    if os.path.exists(x_file[:-9] + '_atlasn.nii.gz') and not overwrite:
        print("Already exists")
        return
    

    transform_fwd_path = os.path.join(transforms_path, x_name + '_fwd.mat')
    transform_inv_path = os.path.join(transforms_path, x_name + '_inv.mat')
    x = ants.image_read(x_file)
    #transform_fwd = ants.read_transform(transform_fwd_path)
    #transform_inv = ants.read_transform(transform_inv_path)

    atlas_transformed = ants.apply_transforms(fixed=x, moving=atlas_aneurysm_smoothed, transformlist=[transform_inv_path])
    x_nib = nib.load(x_file)
    atlas_nib = nib.Nifti1Image(atlas_transformed.numpy(), affine=x_nib.affine)
    assert atlas_nib.shape == x_nib.shape ,print("Shape mismatch",atlas_nib.shape, x_nib.shape)
    nib.save(atlas_nib, os.path.join(data_path, x_name + '_atlasn.nii.gz'))

for x_file, y_file in zip(x_files, y_files):
    transform_and_save_atlas(x_file, data_path, transforms_path_inv, atlas_aneurysm_smoothed, overwrite=True)


# %%

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB


#%%

kernel = 1.0 * RBF(1.0)
gpr = GaussianProcessRegressor(random_state=0)

gnv = GaussianNB()

#%%
atlas_aneurysm_numpy = atlas_aneurysm.numpy() *len(y_files)
input = np.argwhere(atlas_aneurysm_numpy != 0)
output = atlas_aneurysm_numpy[input[:, 0], input[:, 1], input[:, 2]]


# %%
gnv.fit(input, output)
# %%
from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(input)
# %%
kde.score_samples(input[500:5002,:])
# %%
data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias/data/'
all_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
x_files = sorted([f for f in all_files if f.endswith('_x.h5')])
y_files = sorted([f for f in all_files if f.endswith('_y.h5')])
mask_files = sorted([f for f in all_files if f.endswith('_mask.h5')])
import h5py
def read_h5(path):
    reader = h5py.File(path)
    data = reader['data'][()]
    return data
index = 1
mask_path = mask_files[index]
x_path = x_files[index]
y_path = y_files[index]

x,y,mask = read_h5(x_path), read_h5(y_path), read_h5(mask_path)

Z_slice = np.median(np.argwhere(y==4)[:,2]).astype(int)
plt.imshow(x[:, :, Z_slice], cmap='gray')
plt.imshow(y[:, :, Z_slice]==4, alpha=0.5, cmap='jet')
plt.imshow(mask[:, :, Z_slice], alpha=0.2, cmap='jet')
# %%
