import torch
import time


class PostProcessor(object):
    def __init__(self, post_transformations:list, debug:bool = False) -> None:
        self.transforms = post_transformations
        self.debug = debug


    def __call__(self, img:torch.Tensor) -> torch.Tensor:
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

