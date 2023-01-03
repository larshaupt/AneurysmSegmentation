#%%
import torch
import numpy as np
from torchvision import transforms
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
from monai import transforms
import time
import pdb
import torchio as tio
#%%
# =======================
# Transforms the given image and label to pytorch tensor
# =======================
class ToTensor(object):

    def __init__(self, device="cpu") -> None:
        self.device = device

    def __call__(self, sample):

        image, target = sample['image'], sample['target']

        image = np.array(image)
        image = torch.from_numpy(image)
        image = image.type(torch.float32)
        image = image.to(self.device)
        
        target = np.array(target)
        target = torch.from_numpy(target)
        target = target.type(torch.int8)
        target = target.to(self.device)

        sample['image'] = image
        sample['target'] = target
        return sample

class ComposeTransforms:


    def __init__(self, transforms, debug=False):
        self.transforms = transforms
        self.debug = debug

    def __call__(self, img):
        for t in self.transforms:
            if t is not None:
                if self.debug:
                    start_time = time.time()
                    orig_shape = (img["image"].shape, img["target"].shape)
                    if isinstance(img["image"], torch.Tensor) and isinstance(img["target"], torch.Tensor):
                        orig_device = (img["image"].device, img["target"].device)
                    else:
                        orig_device = ("numpy", "numpy")
                    orig_dtype = (img["image"].dtype, img["target"].dtype)
                try:
                    img = t(img)
                except Exception as e:
                    raise Exception("Exception in " + str(t) +" : " + str(e))
                
                if self.debug:
                    new_shape = (img["image"].shape, img["target"].shape)
                    print(f"{t} \n Shape : {orig_shape} --> {new_shape}")
                    if isinstance(img["image"], torch.Tensor) and isinstance(img["target"], torch.Tensor):
                        new_device = (img["image"].device, img["target"].device)
                    else:
                        new_device = ("numpy", "numpy")
                    new_dtype = (img["image"].dtype, img["target"].dtype)
                    print(f"Device : {orig_device} --> {new_device} \n Time: {time.time() - start_time}s \n DType : {orig_dtype} --> {new_dtype} \n")
        return img

# =======================
# Randomly translates the given image and label.
# Translation amounts in x and y axes are given as an array called translate.
# Translation in x axes is done by a value randomly chosen from (-translate[0], translate[0])
# Translation in y axes is done by a value randomly chosen from (-translate[1], translate[1])
# =======================
class Translate(object):
    
    def __init__(self, translate = None):
        self.translate = translate
    
    def __call__(self, sample):
        # print('Translate start')
        translation_x = np.random.randint(-self.translate[0], self.translate[0])
        translation_y = np.random.randint(-self.translate[1], self.translate[1])
        
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        image = np.expand_dims(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])), axis = -1)
        target = np.expand_dims(cv2.warpAffine(target, M, (target.shape[1], target.shape[0])), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)

        if 'seg' in sample:
            seg = np.expand_dims(cv2.warpAffine(seg, M, (seg.shape[1], seg.shape[0])), axis = -1)
            seg = np.squeeze(seg)
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Randomly rotates the given image and label.
# Range for the rotation amount is given as an array called rotate
# and the angle is sampled from the range 0,rotate[0] and 0,rotate[1] and 0,rotate[2]
# =======================
class RandRotate(object):
    
    def __init__(self, rotate_range = [np.pi, np.pi, np.pi], prob=0.1):
        self.rotate_range = rotate_range
        self.prob = prob
        self.rotate = transforms.RandRotate(self.rotate_range[0], self.rotate_range[1], self.rotate_range[2], prob=self.prob)
    
    def __call__(self, sample):

        image, target = sample['image'], sample['target']
        
        target = self.rotate(target, mode= "nearest", randomize=True).as_tensor()
        image = self.rotate(image, mode= "bilinear", randomize=False).as_tensor()
        
        image = np.squeeze(image)
        target = np.squeeze(target)
        
        sample['image'] = image
        sample['target'] = target
        return sample

# =======================
# Randomly flips image and label around x or y axis with same probability
# =======================

class RandomFlip(object):
    def __init__(self, prob = 0.5, spatial_axis=None) -> None:
        self.prob = prob
        self.spatial_axis=spatial_axis
        self.flip = transforms.RandFlip(self.prob, self.spatial_axis)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        image = self.flip(image, randomize=True).as_tensor()
        target = self.flip(target, randomize=False).as_tensor()

        sample['image'] = image
        sample['target'] = target
        return sample

# =======================
# Scales image and label using a random zoom_factor between [0.5, 1.5]
# If zoom_factor < 1.0 zoom_out
# If zoom_factor > 1.0 zoom_in
# =======================
# BUG -- NOT RETURNING 0-1 FOR BINARY IMAGES
# =======================
class ScaleByAFactor(object):
    
    def clipped_zoom(self, img, zoom_factor):

        h, w = img.shape[:2]
    
        # Zooming out
        if zoom_factor < 1:
    
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
    
            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_factor)
    
        # Zooming in
        elif zoom_factor > 1:
    
            out = zoom(img, zoom_factor)
            center_out_h, center_out_w = out.shape[0] // 2, out.shape[1] // 2
            center_img_h, center_img_w = h // 2, w // 2
            top_left_h = center_out_h - center_img_h
            top_left_w = center_out_w - center_img_w
            
            out = out[top_left_h:(top_left_h + h), top_left_w:(top_left_w + w)]
    
        # If zoom_factor == 1, just return the input array
        else:
            out = img
    
        return out
        
    def __call__(self, sample):
        # print('Scalebyafactor start')
        # image, target, seg = sample['image'], sample['target'], sample['seg']
        image, target = sample['image'], sample['target']

        scale = np.random.rand() + 0.5
        image = np.expand_dims(self.clipped_zoom(image, scale), axis = -1)
        target = np.expand_dims(self.clipped_zoom(target, scale), axis = -1)
        # seg = np.expand_dims(self.clipped_zoom(seg, scale), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)
        # seg = np.squeeze(seg)
        
        # return {'image': image, 'target': target, 'seg': seg}
        sample['image'] = image
        sample['target'] = target
        return sample

# =======================
# Randomly crop a region with a specified size from the image and label.
# Size should be specified as an array (output_size)
# where output_size[0] is the height and output_size[1] 
# is the width of the cropped image
# =======================
class CropRandom(object):

    def __init__(self, size, prob=1.0) -> None:
        assert len(size) == 3
        self.size = size
        self.crop = transforms.RandSpatialCrop(self.size, random_size=False)
        self.prob = prob

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        if torch.rand(1).item() < self.prob:
            image = self.crop(image, randomize=True).as_tensor()
            target = self.crop(target, randomize=False).as_tensor()
        sample['image'] = image
        sample['target'] = target
        return sample


# =======================
# Resize the given image and label to a specified size
# Size should be specified as an array (output_size)
# where output_size[0] is the height and output_size[1] 
# is the width of the resized image
# =======================
class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        new_h, new_w = self.output_size[0], self.output_size[1]

        new_h, new_w = int(new_h), int(new_w)
        
        tr = transforms.Compose([transforms.ToPILImage(), transforms.Resize((new_h, new_w)), transforms.ToTensor()])
        image = tr(image).squeeze().numpy()
        target = tr(target).squeeze().numpy()

        if 'seg' in sample:
            seg = (tr(seg).squeeze().numpy() > 0).astype(np.float32)
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Apply Color Jitter (brightness and contrast) to a given image
# =======================
class ColorJitter(object):    
    def __init__(self, brightness = (0.01, 2.0), contrast = (0.01, 2.0)):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        img = torch.from_numpy(image.copy()).float()
        pil_image = transforms.ToPILImage()(img)
        pil_tr = transforms.ColorJitter(brightness = self.brightness, contrast = self.contrast)(pil_image)
        image = np.asarray(pil_tr)

        if 'seg' in sample:
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Apply a set of transformations with some probability
# =======================
class RandomApply(object):
    
    def __init__(self, tf, p):
        self.tf = tf
        self.p = p
    
    def __call__(self, sample):
        
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']

        p_sample = np.random.rand()
        if self.p > p_sample:
            sample = self.tf(sample)
            if 'seg' in sample:
                image, target, seg = sample['image'], sample['target'], sample['seg']
            else:
                image, target = sample['image'], sample['target']
        
        if 'seg' in sample:
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}


class BinarizeSingleLabel(object):
    def __init__(self, label=4):
        self.label = label
    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']

        target = np.where(target==self.label, 1, 0)
        sample['target'] = target
        return sample

class BinarizeSingleLabelTorch(object):
    def __init__(self, label=4):
        self.label = label
    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']
        
        target = torch.where(target==self.label, 1, 0).float()
        sample['target'] = target
        return sample


    

class CollapseLabelsTorch(object):
    def __init__(self, labels_to_collapse:list) -> None:
        self.labels_to_collapse = labels_to_collapse
        self.labels_to_collapse.sort(reverse=True)
        self.target_label = self.labels_to_collapse.pop() # take the lowest element and remove it from list

    def __call__(self, sample) -> dict:
        target = sample['target']
        for label in self.labels_to_collapse:
            target = torch.where(target==label, self.target_label, target)
        sample['target'] = target
        return sample


class TakeNeighboringLabels(object):
    def __init__(self, n_vessel_label = 2, aneurysm_label = 3,n_vessels:dict=None) -> None:

        if n_vessels is None:
            self.n_vessels = {'MCA': [5,6],
                'MA': [5,6],
                'AComm': [8,9,11],
                'ACA': [9,10],
                'Pericallosa': [9,10], # Same as ACA
                'PComm': [1,13,16],
                'BA': [3,16],
                'ICA': [1,5,8],
                'PCA': [3,16,17],
                'PICA': [3,13],
                'VA': [3],
                'PG' : []
                } 
        else:
            self.n_vessels = n_vessels

        self.n_vessel_label = n_vessel_label
        self.aneurysm_label = aneurysm_label

    def __call__(self, sample) -> dict:
        pdb.set_trace()
        target, name = sample['target'], sample['name']
        n_v_cand =  [self.n_vessels[key] for key in self.n_vessels.keys() if key.lower() in name.lower()]
        if len(n_v_cand) == 1:
            n_v = n_v_cand[0]
        elif len(n_v_cand) > 1:
            print(f'TakeNeighboringLabels: Found too many possible aneurysm classes fp {name}: {n_v_cand}') 
            n_v = n_v_cand[0]
        else:
            print(f'TakeNeighboringLabels: Found no possible aneurysm class for {name}')
            n_v = []

        for label in n_v:
            target = torch.where(target==label, self.n_vessel_label, target)

        # remove all other labels than aneurysm and neighboring vessels
        labels_to_keep = [self.n_vessel_label, self.aneurysm_label]
        target = target*torch.sum(torch.stack([target == e for e in labels_to_keep]), dim=0) 
        

        sample['target'] = target
        return sample

class MapLabelTorch(object):
    def __init__(self, label_origin:int, label_target:int) -> None:
        self.label_origin = label_origin
        self.label_target = label_target

    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']
        target = torch.where(target==self.label_origin,  self.label_target, target)
        sample['target'] = target
        return sample


class BinarizeAllLabel(object):
    def __init__(self):
        pass
    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']

        target = np.where(target!= 0, 1, 0)

        sample['target'] = target
        return sample

class BinarizeAllLabelTorch(object):
    def __init__(self):
        pass
    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']

        target = torch.where(target!= 0, 1, 0)

        sample['target'] = target
        return sample

class OneHotEncodeLabel(object):
    def __init__(self, num_classes=1) -> None:
        self.num_classes = num_classes
    def __call__(self, sample) -> dict:

        image, target = sample['image'], sample['target']
        if isinstance(target, torch.Tensor):
            target_new = torch.movedim(torch.nn.functional.one_hot(target.long(), self.num_classes),-1,1).to(torch.bool).squeeze(0)
        elif isinstance(target, np.ndarray):
            target_new = torch.movedim(torch.nn.functional.one_hot(torch.from_numpy(target.astype("long")), self.num_classes),-1,1).squeeze(0)
            target_new = target_new.detach().numpy().astype(bool)

        sample['target'] = target_new
        return sample




class MinMaxNormalizer(object):
    def __init__(self, percentile=99, max_value = 255) -> None:
        self.percentile=percentile
        self.max_value = max_value

    def __call__(self, sample) -> dict:
        image, target = sample['image'], sample['target']
        max_value_img = np.percentile(image, self.percentile)
        min_value_img = np.min(image) #should be 0
        image = (image - min_value_img) * self.max_value / (max_value_img-min_value_img)
        max_mask = np.where(image> self.max_value)
        image[max_mask] = self.max_value

        sample['image'] = image
        sample['target'] = target
        return sample


class Downsample(object):
    def __init__(self, output_size) -> None:
        self.output_size = output_size

    def __call__(self, sample:dict ) -> dict:
        image, target = torch.unsqueeze(sample['image'],0), torch.unsqueeze(sample['target'],0)
        #print(image.shape, self.output_size)
        image = torch.nn.functional.interpolate(image, size = self.output_size, mode='trilinear')
        target =  torch.nn.functional.interpolate(target, size = self.output_size, mode='nearest')

        sample['image'] = torch.squeeze(image,0)
        sample['target'] = torch.squeeze(target,0)
        return sample

class CropSides(object):
    def __init__(self, crop_size) -> None:
        self.crop_size = crop_size

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        old_shape = image.shape[-3:]
        new_shape = []
        for i, dim in enumerate(old_shape):
            new_shape.append(min(self.crop_size[i], dim))

        top = int((old_shape[0]-new_shape[0])/2)
        left = int((old_shape[1]-new_shape[1])/2)
        back = int((old_shape[2]-new_shape[2])/2)
        #print(top, left, back)
        #print(top + new_shape[0],left+ new_shape[1], back+new_shape[2] )

        image = image[...,back: top + new_shape[0], left : left+ new_shape[1], back : back+new_shape[2]]
        target = target[...,back: top + new_shape[0], left : left+ new_shape[1], back : back+new_shape[2]]

        sample['image'] = image
        sample['target'] = target
        return sample
        

class DownsampleByScale(object):
    def __init__(self, scale_factor) -> None:
        self.scale_factor = scale_factor

    def __call__(self, sample:dict ) -> dict:
        image, target = sample['image'], sample['target']
        image, target = torch.unsqueeze(sample['image'],0), torch.unsqueeze(sample['target'],0)
        image = torch.nn.functional.interpolate(image, scale_factor= self.scale_factor, mode='trilinear')
        target =  torch.nn.functional.interpolate(target, scale_factor= self.scale_factor, mode='nearest')


        sample['image'] = torch.squeeze(image,0)
        sample['target'] = torch.squeeze(target,0)
        return sample


class Pad_to(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample:dict) -> dict:
        image, target = torch.unsqueeze(sample['image'],0), torch.unsqueeze(sample['target'],0)
        pad = transforms.SpatialPad((0,*self.size), mode='constant')
        image = pad(image).as_tensor()
        target = pad(target).as_tensor()
        #print(image.shape, self.size)
        sample['image'] = torch.squeeze(image,0)
        sample['target'] = torch.squeeze(target,0)
        return sample


class RandomRotate90(object):
    def __init__(self, prob:float) -> None:
        self.prob = prob
        self.rand_xy = transforms.RandRotate90(prob=self.prob, spatial_axes=(-3,-2), max_k=3)
        self.rand_xz = transforms.RandRotate90(prob=self.prob, spatial_axes=(-3,-1), max_k=3)
        self.rand_yz = transforms.RandRotate90(prob=self.prob, spatial_axes=(-2,-1), max_k=3)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']


        image = self.rand_xy(self.rand_xz(self.rand_yz(image))).as_tensor()
        target = self.rand_xy(self.rand_xz(self.rand_yz(target, randomize=False), randomize=False), randomize=False).as_tensor()


        sample['image'] = image
        sample['target'] = target
        return sample


class CropForeground(object):

    def __init__(self, size, label:int=-1) -> None:
        assert len(size) == 3
        self.size = size
        self.crop = transforms.RandWeightedCrop(spatial_size=self.size, num_samples=1)
        self.label = label

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        if self.label !=  -1:
            if target.shape[0] == 1: 
                # categrorical encoding
                weight_map = (target == self.label)
            else: # self.target.shape[0] > 1
                # one hot encoding
                weight_map = (target[self.label] != 0)
        else: 
            # single binary label
            weight_map = (target >= 0.5)

        image = self.crop(image, weight_map=weight_map, randomize=True)[0].as_tensor()
        target = self.crop(target, weight_map=weight_map, randomize=False)[0].as_tensor()

        sample['image'] = image
        sample['target'] = target
        return sample

class CropForegroundCenter(object):

    def __init__(self, target_label, k_divisible = 16, margin=8, allow_smaller = True) -> None:
        self.target_label = target_label
        self.k_divisible = k_divisible
        self.margin = margin
        def select_target_label(x):
            return x == self.target_label
        self.crop  = transforms.CropForegroundd(keys = ['image', 'target'], source_keys = 'target',select_fn=select_target_label, source_key='target', margin=self.margin, k_divisible=self.k_divisible, allow_smaller=allow_smaller)

    def __call__(self, sample:dict) -> dict:
        sample = self.crop(sample)
        return sample




class CropForegroundBackground(object):
    def __init__(self, size, label:int=-1 ,prob_foreground=0.5) -> None:
        # Samples with probability prob_foreground from foreground
        assert len(size) == 3
        self.size = size
        self.label = label
        self.prob_foreground = prob_foreground
        self.foreground_crop = CropForeground(self.size, self.label)
        self.background_crop = CropRandom(self.size)

    def __call__(self, sample:dict) -> dict:

        if torch.rand(1) < self.prob_foreground:
            cropped_sample = self.foreground_crop(sample)
        else:
            cropped_sample = self.background_crop(sample)
        return cropped_sample


class CropCenter(object):
    def __init__(self, size) -> None:
        assert len(size) == 3
        self.size = size
        self.crop = transforms.CenterSpatialCrop(self.size)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        image = self.crop(image).as_tensor()
        target = self.crop(target).as_tensor()
        sample['image'] = image
        sample['target'] = target
        return sample


class TransformTargetToMulitclass(object):
    def __init__(self) -> None:
        pass

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        target = torch.cat([target, (1-target).to(target.device)], dim=0)
        sample['target'] = target
        return sample
    


class PadToDivisible(object):
    def __init__(self, k=8) -> None:
        self.pad = transforms.DivisiblePad(k)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        image = self.pad(image).as_tensor()
        target = self.pad(target).as_tensor()

        sample['image'] = image
        sample['target'] = target
        return sample

class RandGaussianNoise(object):
    def __init__(self, prob=0.1, std=0.1) -> None:
        self.prob = prob
        self.std = std
        self.noise = transforms.RandGaussianNoise(prob=self.prob, mean=0.0, std=self.std)
    
    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        image = self.noise(image).as_tensor()
        sample['image'] = image

        return sample

class RandElastic(object):

    def __init__(self, sigma_range=(0.1,0.1), magnitude_range=(0.0,0.1), prob=0.1) -> None:
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.prob = prob
        self.elastic = transforms.Rand3DElastic(self.sigma_range, self.magnitude_range, self.prob)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        target = self.elastic(target, mode= "nearest", randomize=True).as_tensor().to(torch.int8)
        image = self.elastic(image, mode= "bilinear", randomize=False).as_tensor()
        sample['image'] = image
        sample['target'] = target
        return sample

class RandAffine(object):
    def __init__(self, scales=(0.9, 1.2), degrees = 15, prob=0.1) -> None:
        self.scales = scales
        self.degrees = degrees
        self.prob = prob
        self.affine = tio.transforms.RandomAffine(scales = self.scales, degrees = self.degrees)

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        if torch.rand(1) < self.prob:
            subject = tio.Subject(image= tio.ScalarImage(tensor=image), target = tio.LabelMap(tensor=target))
            subject = self.affine(subject)
            image, target = subject["image"].data, subject["target"].data

        sample['image'] = image
        sample['target'] = target
        return sample


class CropSidesThreshold(object):

    def __init__(self, smooth = 1) -> None:
        self.smooth = smooth

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        assert image.ndim == 3 or image.ndim == 4
        crop_borders = crop_till_threshold(image, threshold = 1.0, smooth = self.smooth)
        image = crop(image, crop_borders)
        target = crop(target, crop_borders)

        sample['image'] = image
        sample['target'] = target
        return sample


class MaskOutSidesThreshold(object):
    def __init__(self, smooth = 1) -> None:
        self.smooth = smooth

    def __call__(self, sample:dict) -> dict:
        image, target = sample['image'], sample['target']
        crop_borders = crop_till_threshold(image, threshold = 1.0, smooth = self.smooth)
        image = mask_out(image, crop_borders)
        target = mask_out(target, crop_borders)

        sample['image'] = image
        sample['target'] = target
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




# %%
def test_transform(trans):
    a = np.random.random([1,20,20,20])
    b = np.random.random([1,20,20,20]) < 0.5
    
    return trans({"image":a,"target":b})

#ret = test_transform(OneHotEncodeLabel(14))

# %%
img = torch.zeros(40,40,40)
img[10:-10,10:-10,10:-10]=  torch.rand(20,20,20)
target = torch.zeros(40,40,40)
target[10:-10,10:-10,10:-10]=  target = torch.rand(20,20,20) < 0.001


def select_fn(img):
    return img[0] > 0.0
crop = transforms.CropForegroundd(keys = ['image', 'target'], select_fn=select_fn, source_key='target')
#comb = torch.cat([img, target], dim=0).detach().numpy()
comb = {"image":img, "target":target}

ret = crop(comb)

# %%
# %%

# %%
