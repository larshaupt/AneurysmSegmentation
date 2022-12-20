import torch


class MaskOutSidesThreshold(object):
    def __init__(self, smooth = 1) -> None:
        self.smooth = smooth

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        crop_borders = crop_till_threshold(image, threshold = 1.0, smooth = self.smooth)
        image = mask_out(image, crop_borders)
        target = mask_out(target, crop_borders)
        return {'image': image, 'target': target}

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

