#%%
import nibabel as nib
import os
import dicom2nifti
import SimpleITK as sitk
import shutil

#%%

def conv_nrrd2nifti(path:str):
    assert path.endswith(".nrrd")
    head, tail = os.path.split(path)
    if "tof" not in tail:
        tail = "tof_" + tail
    img = sitk.ReadImage(path)
    sitk.WriteImage(img, os.path.join(head, tail.replace(".nrrd", ".nii.gz")))
    sitk.WriteImage(img, path.replace(".nrrd", ".nii.gz"))
    print("Saving to: ",os.path.join(head, tail.replace(".nrrd", ".nii.gz")))

def conv_dicom2nifti(path:str):
    assert os.path.isdir(path)
    new_path = path
    while all([os.path.isdir(el) for el in os.listdir(new_path)]):
        new_path = next(os.listdir(new_path))
    head, tail = os.path.split(path)
    if "tof" not in tail:
        tail = "tof_" + tail
    save_path = head
    dicom2nifti.convert_directory(new_path, save_path)
    print("Saving to: ",head)

def rename_nifti(path:str):
    assert path.endswith(".nii.gz")
    head, tail = os.path.split(path)
    if "tof" not in tail:
        tail = "tof_" + tail
    shutil.copy(path, os.path.join(head, "tof_" + tail))
    print("Saving to: ", os.path.join(head, "tof_" + tail))

#%%
data_path = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_new"

for el in os.listdir(data_path):
    el_path = os.path.join(data_path, el)
    if not os.path.isdir(el_path):
        continue
    #print(el_path)
    all_files = list(os.listdir(el_path))
    tofs = [el for el in all_files if "tof" in el.lower()]

    if len(tofs) == 0 or not any([el.endswith(".nifti") or el.endswith(".nii") or el.endswith(".nii.gz") for el in tofs]):
        
        nrrds = [el for el in tofs if el.endswith("nrrd")]
        if len(nrrds) == 1:
            conv_nrrd2nifti(os.path.join(el_path, nrrds[0]))
            continue
        
        print("\nSelect tof file")
        
        print(all_files)
        tof_file = input()
        print(f"Processing {tof_file}")
        if tof_file.endswith(".nrrd"):
            conv_nrrd2nifti(os.path.join(el_path, tof_file))
        elif os.path.isdir(os.path.join(el_path, tof_file)):
            conv_dicom2nifti(os.path.join(el_path, tof_file))
        elif tof_file.endswith(".nii.gz"):
            rename_nifti(os.path.join(el_path, tof_file))
        else:
            raise Exception(f"File {tof_file} not found in {el_path}")


            


            

# %%
