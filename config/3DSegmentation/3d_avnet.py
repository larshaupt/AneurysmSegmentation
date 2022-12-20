
#####################################################################################################################################################################################################################
import importlib
from data import transformations
from torchvision import transforms as pytorch_tf
import torch.nn as nn
from models.unet_3D import UNet3D, UNet3D_2
import torch
from pytorch_metric_learning import miners, losses

from models.unet import UNet_2 
from utils import DiceMetric, WeightedBCELoss, FocalDiceLoss, DiceCrossEntropyLoss
from torchmetrics import Dice 
from monai.losses import DiceLoss, DiceFocalLoss, FocalLoss
from monai.networks.nets import UNet
from sklearn.metrics import f1_score
import time
importlib.reload(transformations)
# =======================
# Experiment name / input dataset
# =======================
dataset = 'USZ_hdf5'
fold_id = 0
timestamp = int(time.time())
experiment_name = '%s_experiment_avnet_%d'%(dataset, timestamp)

# ==================
# Hyperparameters
# ==================
batch_size = 2
batch_size_val = 2
num_classes = 2
number_of_epoch = 1000
learning_rate = 1e-3
lambda_loss = 1.0



#model = UNet3D_2(1,1)
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3, 
    up_kernel_size=3, 
    num_res_units=0, 
    act='PRELU',
    norm='INSTANCE',
    dropout=0.0,
    bias=True,
    adn_ordering='NDA',
    dimensions=None
) 
"""
model = UNet_2(            
            in_channels = 1,
            out_channels = 1,
            n_blocks = 3,
            start_filts = 4,
            up_mode= 'transpose',
            merge_mode = 'concat',
            planar_blocks = (),
            batch_norm = 'unset',
            attention = False,
            activation = 'relu',
            normalization = 'batch',
            full_norm = True,
            dim = 3)
"""

optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate, weight_decay=1e-4)
#criterion_loss = WeightedBCELoss(weight=1000)
#criterion_loss = DiceLoss(include_background=True, sigmoid=True)
#criterion_loss = WeightedFocalLoss(weight=1e4, sigmoid=True)
#criterion_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([100]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
criterion_loss = DiceCrossEntropyLoss(ratioce=0.5, cepositiveweight=0.99)
#criterion_loss = FocalDiceLoss(gamma=2)
#criterion_loss = FocalLoss()

""" 
miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.ALL,
        neg_strategy=miners.BatchEasyHardMiner.HARD,
        allowed_pos_range=None,
        allowed_neg_range=None,
    ) """


criterion_metric = f1_score


# ==============
# Paths
# ==============

path_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/data/'%(dataset)
path_split = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/k_fold_split2_val.json'%(dataset)

shuffle_train = True
shuffle_validation = False
shuffle_test = False

training_dataset = 'train'
validation_dataset = 'val'
test_dataset = 'test'

num_training_files = -1

path_to_save_pretrained_models = './pre_trained/%s/%s'%(dataset, experiment_name)

# ======================================
# Transformations for data augmentation
# ======================================
tf_train = pytorch_tf.Compose([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.DownsampleByScale((2.0/3.0, 2.0/3.0, 1.0)),  
                transformations.Pad_to((512, 512, 256)),
                transformations.CropForegroundBackground((128,128,128), 0.75),
                #transformations.CropRandom((128,128,128)), 
                transformations.RandomRotate90(0.5),
                transformations.RandomFlip(0.5, (-1,-2,-3)),
                transformations.TransformTargetToMulitclass()
                ])

tf_test = pytorch_tf.Compose([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.DownsampleByScale((2.0/3.0, 2.0/3.0, 1.0)),  
                transformations.Pad_to((512, 512, 256)),
                transformations.CropForegroundBackground((128,128,128), 0.5),
                #transformations.CropRandom((128,128,128)),
                transformations.TransformTargetToMulitclass(),
                ])

#####################################################################################################################################################################################################################