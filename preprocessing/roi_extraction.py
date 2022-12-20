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


#%%

mri_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
out_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data'
# %%

#skull_stripping('/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/10576901_KJ_ICA_x.nii.gz', out_dir)

#%%
save_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_be_nifti/data'
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
    mask = nib.load(os.path.join(out_dir, os.path.basename(x) + "_mask.nii.gz"))
    #mask = extract_brain(image)
    #nib.save(mask, os.path.join(save_path, os.path.basename(y).replace('_y', '_mask')))
    label_data, mask_data = label.get_fdata(), mask.get_fdata()
    lost_points = np.argwhere(np.logical_and(mask_data < 0.5, label_data!=0))
    lost_points_target = np.argwhere(np.logical_and(mask_data > 0.5, label_data==4))
    lost_points_all.append(lost_points)
    lost_points_target_all.append(lost_points_target)
    print(x)
    print(len(lost_points), len(lost_points_target))


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
