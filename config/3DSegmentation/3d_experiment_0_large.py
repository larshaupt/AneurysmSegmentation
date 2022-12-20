
#####################################################################################################################################################################################################################
import importlib
from data import transformations
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import time

from train.metrics import *
from train.losses import *


# =======================
# Experiment name / input dataset
# =======================
dataset = 'USZ_hdf5'
fold_id = 0
k_fold = False
k_fold_k = 5
timestamp = int(time.time())
experiment_name = '%s_experiment_large_%d'%(dataset, timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 1

# ==================
# Hyperparameters
# ==================
batch_size = 1
batch_size_val = 1
num_classes = 1
number_of_epoch = 300
learning_rate = 1e-3
lambda_loss = 1.0
tags = ["large"]
debug=False


model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
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


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)
early_stopping = False
lr_scheduler = 'cycle'
patience = 50

positive_weight = 100
#criterion_loss = WeightedBCELoss(weight=positive_weight)
#criterion_loss = DiceLoss(include_background=True, sigmoid=True)
#criterion_loss = WeightedFocalLoss(weight=1e4, sigmoid=True)
criterion_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
#criterion_loss = DiceCrossEntropyLoss(ratioce=0.5, cepositiveweight=0.99)
#criterion_loss = FocalDiceLoss(gamma=2)
#criterion_loss = FocalLoss()
#criterion_loss = nn.BCEWithLogitsLoss()
#criterion_loss = MixLoss([nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight]).to("cuda:0" if torch.cuda.is_available() else "cpu")), DiceLoss(include_background=True, sigmoid=True)], [0.9,0.1])


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
tf_train = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.DownsampleByScale((0.5/3.0, 0.5/3.0, 0.5)),
                transformations.PadToDivisible(16),
                ])

tf_val = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.DownsampleByScale((0.5/3.0, 0.5/3.0, 0.5)),
                transformations.PadToDivisible(16),
                ])

tf_test = transformations.ComposeTransforms([
                transformations.BinarizeSingleLabel(), 
                transformations.ToTensor(), 
                transformations.DownsampleByScale((0.5/3.0, 0.5/3.0, 0.5)),
                transformations.PadToDivisible(16)
                ])

#####################################################################################################################################################################################################################