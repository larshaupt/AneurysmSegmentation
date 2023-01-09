
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
import ants
import shutil

save_path = '/srv/beegfs02/scratch/brain_artery/data/training/affines_inv'
atlas_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas'
atlas_tof_path = os.path.join(atlas_path, 'tofAverage.nii.gz')

vessel_path = os.path.join(atlas_path, 'vesselProbabilities.nii.gz')
radius_path = os.path.join(atlas_path, 'vesselRadius.nii.gz')
#data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/ADAM_Challenge/ADAM_release_subjs'
data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data'
img_fix = ants.image_read(atlas_tof_path)


all_files = [os.path.join(data_path,f ) for f in list(os.listdir(data_path))]
x_files = [f for f in all_files if f.endswith('_x.nii.gz')]

#x_files = [os.path.join(data_path, f, "pre", "TOF.nii.gz") for f in os.listdir(data_path)]
#x_files = sorted(x_files)


def compute_registration(x_path, save_path, img_atlas, data_path, save_plot=True, overwrite=False, deformable=False):
    #x_name = os.path.basename(x_path[:-15])
    x_name = os.path.basename(x_path[:-9])

    file_save_path = os.path.join(save_path, x_name)
    print(file_save_path + '_fwd.mat')

    if os.path.exists(file_save_path + '_fwd.mat') and not overwrite:
        print('Already computed')
        return


    img_fix = ants.image_read(x_path)
    img_move = img_atlas
    
    ## Parameters ##

    ## Registration ##


    mytx_syn = ants.registration(fixed=img_fix, moving=img_move, 
        type_of_transform="Affine",
        aff_metric = 'mattes',
        syn_metric = 'mattes')
    print(mytx_syn['fwdtransforms'])
    transformed = ants.apply_transforms(fixed=img_fix, moving=img_move, transformlist=mytx_syn['fwdtransforms'])
    
    if save_plot:
        
        regtools.overlay_slices(img_fix.view(), transformed.view(), None, 0,"Template", "Moving", fname=file_save_path + '_overlay0.png')
        regtools.overlay_slices(img_fix.view(), transformed.view(), None, 1,"Template", "Moving", fname=file_save_path + '_overlay1.png')
        regtools.overlay_slices(img_fix.view(), transformed.view(), None, 2,"Template", "Moving", fname=file_save_path + '_overlay2.png')



    ants.write_transform(ants.read_transform(mytx_syn['fwdtransforms'][0]), file_save_path + '_fwd.mat')
    ants.write_transform(ants.read_transform(mytx_syn['invtransforms'][0]), file_save_path + '_inv.mat')
    
    if deformable:
        shutil.move(mytx_syn['fwdtransforms'][1], file_save_path + '_fwd.nii.gz')
        shutil.move(mytx_syn['invtransforms'][1], file_save_path + '_inv.nii.gz')



    if os.path.exists(mytx_syn['fwdtransforms'][0]):
        os.remove(mytx_syn['fwdtransforms'][0])
    if os.path.exists(mytx_syn['invtransforms'][0]):
        os.remove(mytx_syn['invtransforms'][0])
    if deformable:
        if os.path.exists(mytx_syn['fwdtransforms'][1]):
            os.remove(mytx_syn['fwdtransforms'][1])
        if os.path.exists(mytx_syn['invtransforms'][1]):
            os.remove(mytx_syn['invtransforms'][1])

    return None

def run_process(every_n = 4, start_i = 0):
    for i in range(start_i, len(x_files), every_n):
        compute_registration(x_files[i], save_path, img_fix, data_path, save_plot=True, overwrite=False)
#run_process(1,0)

ps = []
n = 3
split_dif = n
split_id = 0
for k in range(split_id*split_dif, split_dif*(split_id+1)):
    ps.append(multiprocessing.Process(target=run_process, args = (n,k,)))

for k in range(len(ps)):
    ps[k].start()

for k in range(len(ps)):
    ps[k].join()
