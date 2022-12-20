
#####################################################################################################################################################################################################################
import importlib
from data import transformations
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import torch
from monai.networks.nets import UNet, AttentionUnet, UNETR, AHNet
import time

from train.metrics import *
from train.losses import *


# =======================
# Experiment name / input dataset
# =======================
dataset = 'USZ_hdf5d2'
fold_id = 0
project_name = "USZ_opt"
k_fold = False
k_fold_k = 5
timestamp = int(time.time())
experiment_name = '%s_experiment_simple_%d'%(dataset, timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 4


# ==================
# Hyperparameters
# ==================
batch_size = 2
batch_size_val = 1
num_classes = 1
number_of_epoch = 300
learning_rate = 1e-3
lambda_loss = 1.0
tags = ["simple", "opt"]
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

""" model = AttentionUnet(
    spatial_dims = 3, 
    in_channels = 1, 
    out_channels = num_classes, 
    channels = (16, 32, 64, 128, 256), 
    strides = (2, 2, 2, 2), 
    kernel_size=3, 
    up_kernel_size=3, 
    dropout=0.0) """
pretrained_weights = ""
weight_decay = 0
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
early_stopping = False
lr_scheduler = 'reduce'
patience = 50

positive_weight = 100
#criterion_loss = WeightedBCELoss(weight=positive_weight)
#criterion_loss = DiceLoss()
#criterion_loss = WeightedFocalLoss(weight=1e4, sigmoid=True)
criterion_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
#criterion_loss = DiceCrossEntropyLoss(ratioce=0.5, cepositiveweight=0.99)
#criterion_loss = FocalDiceLoss(gamma=2)
#criterion_loss = FocalLoss()
#criterion_loss = nn.BCEWithLogitsLoss()
#criterion_loss = MixLoss([nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight]).to("cuda:0" if torch.cuda.is_available() else "cpu")), DiceLoss(include_background=False, sigmoid=True)], [0.9,0.1])


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

path_to_save_pretrained_models = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/%s'%(experiment_name)



# ======================================
# Transformations for data augmentation
# ======================================
patch_size = (192, 192, 92)
validate_whole_vol = False
augment_probability = 0.1
only_foreground = True
extra_cropping=False
tf_train = transformations.ComposeTransforms([
                transformations.ToTensor(),  
                transformations.BinarizeSingleLabelTorch(label=4), 
                transformations.CropForeground(patch_size) if only_foreground else transformations.CropForegroundBackground(patch_size,augment_probability),
                transformations.CropRandom((np.array(patch_size)/2).astype(int), prob= augment_probability) if extra_cropping else None,
                transformations.PadToDivisible(16),
                transformations.RandomRotate90(prob=augment_probability) if patch_size.count(patch_size[0]) == len(patch_size) else None,
                transformations.RandomFlip(prob=augment_probability),
                transformations.RandElastic(sigma_range=(5,7), magnitude_range=(10,50), prob=augment_probability),
                transformations.RandGaussianNoise(prob=0.1, std=augment_probability)
                ], debug=False)

tf_val = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(label=4),
                transformations.ToTensor(),
                transformations.CropForeground(patch_size) if not validate_whole_vol else None,
                transformations.PadToDivisible(16),
                ])

tf_test = transformations.ComposeTransforms([
                transformations.ToTensor(),
                transformations.BinarizeSingleLabelTorch(label=4),
                transformations.PadToDivisible(16)
                ])

#####################################################################################################################################################################################################################