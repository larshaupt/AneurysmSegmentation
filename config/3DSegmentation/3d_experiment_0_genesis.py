
#####################################################################################################################################################################################################################
import importlib
from data import transformations
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import torch
from monai.networks.nets import UNet, AttentionUnet, UNETR, AHNet
from monai.losses import DiceLoss
import time

from models.unet3d import UNet3D
from train.metrics import *
from train.losses import *



# =======================
# Experiment name / input dataset
# =======================
dataset = 'USZ_hdf5d'
fold_id = 0
k_fold = False
k_fold_k = 5
timestamp = int(time.time())
experiment_name = '%s_experiment_genesis_%d'%(dataset, timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 1


# ==================
# Hyperparameters
# ==================
batch_size = 4
batch_size_val = 4
num_classes = 1
number_of_epoch = 1000
learning_rate = 1e-4
lambda_loss = 1.0
tags = ["genesis"]
debug=False

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
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

pretrained_weights = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Training/pre_trained/USZ_hdf5d_experiment_dae_1669027098/0/model_902.pth'

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)
early_stopping = True
lr_scheduler = 'reduce'
patience = 50

positive_weight = 100

criterion_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
#criterion_loss = DiceCoefLoss()

# ==============
# Metrics
# ==============

criterion_metric =  {"Dice":DiceMetric(),
                    "Recall": RecallMetric(),
                    "Precision": PrecisionMetric(),
                    }



# ==============
# Paths
# ==============

path_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/data/'%(dataset)
path_split = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/k_fold_split2_val.json'%(dataset)


# ==============
# Data Loading and Saving
# ==============

shuffle_train = True
shuffle_validation = False
shuffle_test = False

training_dataset = 'train'
validation_dataset = 'val'
test_dataset = 'test'

num_training_files = -1

path_to_save_pretrained_models = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Training/pre_trained/%s'%(experiment_name)



# ======================================
# Transformations for data augmentation
# ======================================
patch_size = (96,96,96)
augment_probability = 0.1
validate_whole_vol = True
tf_train = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(),  
                transformations.CropForegroundBackground(patch_size, 0.75),
                transformations.RandomRotate90(prob=augment_probability) if patch_size.count(patch_size[0]) == len(patch_size) else None,
                transformations.RandomFlip(prob=augment_probability),
                transformations.RandElastic(sigma_range=(5,7), magnitude_range=(10,50), prob=augment_probability),
                transformations.RandGaussianNoise(prob=0.1, std=augment_probability)
                ], debug=False)

tf_val = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.CropForeground(patch_size) if not validate_whole_vol else None,
                transformations.PadToDivisible(16),
                ])

tf_test = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.PadToDivisible(16)
                ])

#####################################################################################################################################################################################################################