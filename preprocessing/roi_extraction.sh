cd /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data
for FILE in *; do
if [[ "${FILE:0-9}" == "_x.nii.gz" ]]; then
    mri_synthstrip -i "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_nifti/data/${FILE}" -o "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data/${FILE}_mask.nii.gz" -m  "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_bias_fs_nifti/data/${FILE}_mask_m.nii.gz"
fi
done