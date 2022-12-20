
#####################################################################################################################################################################################################################
import importlib
from data import transformations
from data import genesis_transformations
from torchvision import transforms as pytorch_tf
import torch.nn as nn
import torch
from monai.networks.nets import UNet, AttentionUnet, UNETR, AHNet
from monai.losses import DiceLoss
import time

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
experiment_name = '%s_experiment_dae_%d'%(dataset, timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================
# Hyperparameters
# ==================
batch_size = 4
batch_size_val = 4
num_classes = 1
number_of_epoch = 10000
learning_rate = 1e-2
lambda_loss = 1.0
tags = ["dae"]
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

""" model = AHNet(
    layers=(3, 4, 6, 3), 
    spatial_dims=3, 
    in_channels=1, 
    out_channels=1, 
    psp_block_num=4, 
    upsample_mode='transpose', 
    pretrained=True, 
    progress=True) """

""" model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3, 
    up_kernel_size=3, 
    dropout=0.0
) """
""" model = UNETR(
    spatial_dims=3,
    in_channels = 1,
    out_channels = num_classes,
    img_size = (96,96,96),
    feature_size=16,
) """
pretrained_weights = ""

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)
#optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=0.0, nesterov=False)
early_stopping = False
lr_scheduler = 'step'
patience = 50

positive_weight = 300

criterion_loss = nn.MSELoss()



# ==============
# Metrics
# ==============

criterion_metric =  {
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
patch_size = 96
augment_probability = 0.1
tf_train = transformations.ComposeTransforms([
                transformations.ToTensor(), 
                #transformations.DownsampleByScale((2.0/3.0, 2.0/3.0, 1.0)),  
                transformations.RandomCrop((96,96,96)),
                genesis_transformations.GenesisTransform()
                ], debug=False)

tf_val = transformations.ComposeTransforms([
                transformations.ToTensor(), 
                #transformations.DownsampleByScale((2.0/3.0, 2.0/3.0, 1.0)), 
                transformations.RandomCrop((96,96,96)),
                genesis_transformations.GenesisTransform()
                ])

tf_test = transformations.ComposeTransforms([
                transformations.ToTensor(), 
                #transformations.DownsampleByScale((2.0/3.0, 2.0/3.0, 1.0)), 
                ])

#####################################################################################################################################################################################################################