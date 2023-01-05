import nibabel as nib
import nilearn
from nilearn.plotting import plot_anat, plot_roi
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)
from dipy.align import affine_registration
import os
import pickle
import numpy as np
import multiprocessing


save_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/affines'
atlas_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas'
atlas_tof_path = os.path.join(atlas_path, 'tofAverage.nii.gz')

vessel_path = os.path.join(atlas_path, 'vesselProbabilities.nii.gz')
radius_path = os.path.join(atlas_path, 'vesselRadius.nii.gz')
data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'

template_img = nib.load(atlas_tof_path)
template_data = template_img.get_fdata()
template_affine = template_img.affine

all_files = list(os.listdir(data_path))
x_files = [f for f in all_files if f.endswith('_x.nii.gz')]

x_files = sorted(x_files)



def compute_registration(x_name, save_path, template_data, template_affine, data_path, save_plot=True, overwrite=False):

    print(x_name)

    if os.path.exists(os.path.join(save_path, x_name[:-9] + '_affine.npy')) and not overwrite:
        print('Already computed')
        return
        
    img_path = os.path.join(data_path, x_name)

    image = nib.load(img_path)
    image_data = image.get_fdata()
    image_affine = image.affine

    moving_data, moving_affine = image_data, image_affine
    static_data, static_affine = template_data, template_affine


    ## Parameters ##
    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    level_iters = [10000, 1000, 100]

    ## Registration ##

    xformed_img, reg_affine = affine_registration(
        moving_data,
        static_data,
        moving_affine=moving_affine,
        static_affine=static_affine,
        nbins=32,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)

    np.save(os.path.join(save_path, x_name[:-9] + '_affine.npy'), reg_affine)
    
    affine_transform = AffineMap(reg_affine,template_data.shape, template_affine,moving_data.shape, moving_affine)
    transformed = affine_transform.transform(moving_data)

    if save_plot:
        regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Moving", fname=os.path.join(save_path, x_name[:-9] + '_overlay0.png'))
        regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Moving", fname=os.path.join(save_path, x_name[:-9] + '_overlay1.png'))
        regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Moving", fname=os.path.join(save_path, x_name[:-9] + '_overlay2.png'))

    return reg_affine


def run_process(every_n = 4, start_i = 0):
    for i in range(start_i, len(x_files), every_n):
        compute_registration(x_files[i], save_path, template_data, template_affine, data_path, save_plot=True)

ps = []
n = 2
split_dif = n
split_id = 0
for k in range(split_id*split_dif, split_dif*(split_id+1)):
    ps.append(multiprocessing.Process(target=run_process, args = (n,k,)))

for k in range(len(ps)):
    ps[k].start()

for k in range(len(ps)):
    ps[k].join()