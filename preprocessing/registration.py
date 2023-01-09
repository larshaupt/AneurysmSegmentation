#%%
import nibabel as nib
import nilearn
from nilearn.plotting import plot_anat, plot_roi
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)
import os
from preprocessing_utils import *
import SimpleITK as sitk
import sys
import ants


atlas_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Atlas'
atlas_tof_path = os.path.join(atlas_path, 'tofAverage.nii.gz')
img_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/02053152-MCA-rechts-NEW_x.nii.gz'
target_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/02053152-MCA-rechts-NEW_x.nii.gz'
vessel_path = os.path.join(atlas_path, 'vesselProbabilities.nii.gz')
radius_path = os.path.join(atlas_path, 'vesselRadius.nii.gz')
#%%

template_img = nib.load(atlas_tof_path)
moving_img = nib.load(img_path)
vessel_img = nib.load(vessel_path)
target_img = nib.load(target_path)
radius_img = nib.load(radius_path)

moving_data = moving_img.get_fdata()
template_data = template_img.get_fdata()
moving_affine = moving_img.affine
template_affine = template_img.affine
vessel_data = vessel_img.get_fdata()
vessel_affine = vessel_img.affine
target_data = target_img.get_fdata()
target_affine = target_img.affine
radius_data = radius_img.get_fdata()
radius_affine = radius_img.affine

# %%
identity = np.eye(4)
affine_map = AffineMap(identity,template_data.shape, template_affine,moving_data.shape, moving_affine)
resampled = affine_map.transform(moving_data)
regtools.overlay_slices(template_data, resampled, None, 0,"Template", "Moving")
regtools.overlay_slices(template_data, resampled, None, 1,"Template", "Moving")
regtools.overlay_slices(template_data, resampled, None, 2,"Template", "Moving")

# %%
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

# %%

affreg = AffineRegistration(metric=metric,level_iters=level_iters,sigmas=sigmas,factors=factors)
# %%
transform = TranslationTransform3D()
params0 = None
translation  = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine)
# %%
print(translation.affine)
transformed = translation.transform(moving_data)
regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Moving")

# %%
transform = RigidTransform3D()
rigid = affreg.optimize(template_data, moving_data, transform, params0, template_affine, moving_affine, starting_affine=translation.affine)


# %%
print(rigid.affine)
transformed = rigid.transform(moving_data)
regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Moving")


# %%

affreg.level_iters = [1000, 1000, 100]
affine = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine,starting_affine=rigid.affine)

# %%

print(affine.affine)
transformed = affine.transform(moving_data)
regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Moving")
regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Moving")

# %%
transformed = affine.transform(moving_data)
regtools.overlay_slices(vessel_data, transformed, None, 0,"Template", "Moving")
regtools.overlay_slices(vessel_data, transformed, None, 1,"Template", "Moving")
regtools.overlay_slices(vessel_data, transformed, None, 2,"Template", "Moving")

# %%


plot_anat(moving_img, title='Moving Image')
# %%
vessel_tr = affine.transform_inverse(vessel_data)
radius_tr = affine.transform_inverse(radius_data)
# %%
plt.imshow(moving_data[:,:,100].T, cmap='gray', origin='lower')
plt.imshow(vessel_tr[:,:,100].T, cmap='jet', alpha=0.3, origin='lower')
plt.imshow(target_data[:,:,100].T, cmap='rainbow', alpha=0.3, origin='lower')
# %%
index = 100
mask = (vessel_tr > 1) * (moving_data > 100) * (radius_tr > 0.5)
plt.imshow((moving_data*mask)[:,:,100], cmap='gray', origin='lower')
plt.imshow((target_data*~mask)[:,:,100], cmap='rainbow', alpha=0.8, origin='lower')
 # %%


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


if len(sys.argv) < 4:
    print(
        "Usage:",
        sys.argv[0],
        "<fixedImageFilter> <movingImageFile>",
        "<outputTransformFile>",
    )
    sys.exit(1)

fixed = sitk.ReadImage(img_path, sitk.sitkFloat32)

moving = sitk.ReadImage(vessel_path, sitk.sitkFloat32)

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

sitk.WriteTransform(outTx, sys.argv[3])

if "SITK_NOSHOW" not in os.environ:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    sitk.Show(cimg, "ImageRegistration1 Composition")

# %%

img_fix = ants.image_read(atlas_tof_path)
img_move = ants.image_read(img_path)
mytx_syn = ants.registration(fixed=img_fix, moving=img_move, 
    type_of_transform="Affine",
    aff_metric = 'mattes',
    syn_metric = 'mattes')
# %%


transformed = ants.apply_transforms(fixed=img_move, moving=img_fix, transformlist=mytx_syn['fwdtransforms'])


regtools.overlay_slices(img_move.view(), transformed.view(), None, 0,"Template", "Moving")
regtools.overlay_slices(img_move.view(), transformed.view(), None, 1,"Template", "Moving")
regtools.overlay_slices(img_move.view(), transformed.view(), None, 2,"Template", "Moving")

# %%