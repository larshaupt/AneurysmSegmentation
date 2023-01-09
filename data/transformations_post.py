import torch
import numpy as np
import cc3d
import train.utils as ut

class Binarize(object):
    def __init__(self, threshold = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, sample:dict) -> dict:
        pred = sample['pred']
        pred = ut.binarize(pred, threshold = self.threshold)
        sample["pred"] = pred
        return sample


class MaskOutSidesThreshold(object):
    def __init__(self, smooth = 1) -> None:
        self.smooth = smooth

    def __call__(self, sample:dict) -> dict:
        image, pred, mask = sample['image'], sample['pred'], sample['mask']
        crop_borders = crop_till_threshold(image, threshold = 1.0, smooth = self.smooth)
        image = mask_out(image, crop_borders)
        pred = mask_out(pred, crop_borders)
        mask = mask_out(mask, crop_borders)
        sample["image"] = image
        sample["pred"] = pred
        sample["mask"] = mask
        return sample

def crop(img, crop_borders):
    assert img.ndim == 3 or img.ndim == 4
    orig_shape = img.shape
    img = img.view(-1, *orig_shape[-3:])
    img = img[:, crop_borders["x"][0]:crop_borders["x"][1], crop_borders["y"][0]:crop_borders["y"][1], crop_borders["z"][0]:crop_borders["z"][1]]
    if len(orig_shape) == 3:
        return img.squeeze(0)
    else:
        return img

def mask_out(img, crop_borders):
    assert img.ndim == 3 or img.ndim == 4
    orig_shape = img.shape
    img = img.view(-1, *orig_shape[-3:])

    img_new = torch.zeros_like(img)
    img_new[:, crop_borders["x"][0]:crop_borders["x"][1], crop_borders["y"][0]:crop_borders["y"][1], crop_borders["z"][0]:crop_borders["z"][1]] = \
        img[:, crop_borders["x"][0]:crop_borders["x"][1], crop_borders["y"][0]:crop_borders["y"][1], crop_borders["z"][0]:crop_borders["z"][1]]
        
    if img.shape[0] > 1: # multiclass labels
        # additionally, set the background label to 0 if masked out
        img_new[0, crop_borders["x"][0]:crop_borders["x"][1], crop_borders["y"][0]:crop_borders["y"][1], crop_borders["z"][0]:crop_borders["z"][1]] = 1

    return img.view(*orig_shape)


def find_first_threshold_loc(img, threshold,  axis = 0 ,smooth=1 , backwards=False):
    assert axis in [0,1,2]
    assert img.ndim == 3
    if backwards==False:
        iter_list = list(range(img.shape[axis]))
    else:
        iter_list = list(range(img.shape[axis]))[::-1]
    if axis == 0:
        for loc in iter_list:
            if torch.sum(img[loc, :,:] > threshold) >= smooth:
                return loc
    elif axis == 1:
        for loc in iter_list:
            if torch.sum(img[:, loc,:] > threshold) >= smooth:
                return loc
    elif axis == 2:
        for loc in iter_list:
            if torch.sum(img[:,:, loc] > threshold) >= smooth:
                return loc

    # If the slice is not found, we just return the original dimensions
    if backwards:
        return img.shape[axis]
    return 0

def crop_till_threshold(img, threshold, axes:list = ["x", "y", "z"], smooth = 1):
    img = img.view(*img.shape[-3:]) # Just take the last three dimensions of the image
    threshold_locs = {}
    for ax in axes:
        if ax == "x":
            threshold_locs[ax] = (find_first_threshold_loc(img, threshold, axis=0, smooth=smooth,backwards=False), find_first_threshold_loc(img, threshold, axis=0, smooth=smooth,backwards=True))
        elif ax =="y":
            threshold_locs[ax] = (find_first_threshold_loc(img, threshold, axis=1, smooth=smooth,backwards=False), find_first_threshold_loc(img, threshold, axis=1, smooth=smooth,backwards=True))
        elif ax =="z":
            threshold_locs[ax] = (find_first_threshold_loc(img, threshold, axis=2, smooth=smooth,backwards=False), find_first_threshold_loc(img, threshold, axis=2, smooth=smooth,backwards=True))

    return threshold_locs


def remove_small_components(labels, thr=100, thr_upper=0):

            
    if labels.shape[0] == 3:
        labels = labels[2,:,:,:].unsqueeze(0)
    elif labels.shape[0] == 22:
        labels = labels[4,:,:,:].unsqueeze(0)

    orig_shape = labels.shape
    if isinstance(labels, torch.Tensor):
        torch_flag = True
        labels = labels.detach().numpy()
    else:
        torch_flag = False
    assert labels.ndim in [3,4,5], 'wrong input shape, too many or too few dimensions'
    assert issubclass(labels.dtype.type, np.integer) or issubclass(labels.dtype.type, np.bool8), f'Wrong input type. Need to be int. Found: {labels.dtype}'
    orig_shape = labels.shape
    orig_dtype = labels.dtype

    labels = np.reshape(labels, labels.shape[-3:])

    
    labels_in = cc3d.connected_components(labels.astype(int), connectivity=26, return_N=False)
    
    stat = cc3d.statistics(labels_in)
    cc_size = stat['voxel_counts']

    
    if thr_upper is not None and thr_upper > thr:
        vals_to_keep = np.argwhere((cc_size > thr) & (cc_size < thr_upper)).flatten()
    else:
        vals_to_keep = np.argwhere(cc_size > thr).flatten()

    inds = labels_in == vals_to_keep[:, None, None, None]
    labels[~np.any(inds, axis = 0)] = 0

    

    labels = np.reshape(labels, orig_shape)
    labels = labels.astype(orig_dtype)
    
    if torch_flag:
        labels = torch.from_numpy(labels)

    return labels > 0


class Threshold_cc(object):
    def __init__(self, threshold:int = 100, threshold_upper = 0, thredhold_pred = 0.5) -> None:
        self.threshold = threshold
        self.thredhold_pred = thredhold_pred
        self.threshold_upper = threshold_upper

    def __call__(self, sample:dict) -> dict:
        image, pred = sample['image'], sample['pred']


        thr_mask = remove_small_components(ut.binarize(pred) ,thr_upper = self.threshold_upper, thr=self.threshold)
        pred = pred * thr_mask
        sample["image"] = image
        sample["pred"] = pred
        return sample

class Threshold_data(object):
    def __init__(self, threshold:float = 1.0) -> None:
        self.threshold = threshold

    def __call__(self, sample:dict) -> dict:
        image, pred = sample['image'], sample['pred']
        mask = image > self.threshold
        pred = pred * mask
        sample["image"] = image
        sample["pred"] = pred
        return sample


class Mask_Concentation(object):
    def __init__(self, threshold:float = 1.0) -> None:
        self.threshold = threshold

    def __call__(self, sample:dict) -> dict:
        if 'mask' not in sample.keys():
            print("No mask found in sample")
            return sample
        image, pred, mask = sample['image'], sample['pred'], sample['mask']
        
        mask_th = (mask/torch.max(mask)) > self.threshold
        pred = pred * mask_th

        sample["image"] = image
        sample["pred"] = pred
        return sample