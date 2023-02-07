import argparse
import os
import numpy as np
import time
import torch
import monai
from monai.networks.nets import UNet, AttentionUnet, UNETR, AHNet
import json
from importlib.machinery import SourceFileLoader
from argparse import Namespace
from types import ModuleType
import shutil
import copy
import pdb

from config import PATH_TO_DATASET, PATH_TO_PRETRAINED

from train.metrics import *
from train.losses import *
from train.utils import *
from data import transformations, transformations_post

"""
A class for handling the options and configuration for the main program.

Parameters
----------
mode : str, optional
    The mode of the program, either 'train' or 'test', by default 'train'
verbose : bool, optional
    Whether to print the output in a verbose mode, by default True
config_file : str, optional
    The configuration file, either a '.py' or a '.json' file, by default ''
**kwargs : optional
    Other arguments, will be stored as self.kwargs
    
Attributes
----------
kwargs : dict
    Other arguments passed
opt : Namespace
    Namespace object for storing the options and configuration
inp_dict : dict
    Dictionary object for storing the options from the configuration file
file_cont : str
    String object for storing the content of the configuration file
verbose : bool
    Whether to print the output in a verbose mode
config_file : str
    The configuration file, either a '.py' or a '.json' file
source : str
    The source of the options and configuration, either 'py', 'json', or 'args'
"""


class Options:

   
    def __init__(self, mode:str = 'train', 
                verbose:bool = True, 
                config_file:str = '',
                **kwargs):


        self.kwargs = kwargs
        self.opt = Namespace()
        self.inp_dict = dict()
        self.file_cont = ""
        self.verbose = verbose
        

        if config_file == '':
            self.parse_arguments()
            self.config_file = self.inp.exp_path
        else:
            self.config_file = config_file
            
        if self.config_file != '' and self.config_file.endswith("py"):
            self.source = "py"
            if self.verbose:
                print(f'\n#####################################################\nReading the configuration from python file: \n{self.config_file}\n#####################################################\n')
        elif self.config_file != '' and self.config_file.endswith("json"):
            self.source = "json"
            if self.verbose:
                print(f'\n#####################################################\nReading the configuration from json file: \n{self.config_file}\n#####################################################\n')
        else:
            self.source = "args"
            if self.verbose:
                print(f'\n#####################################################\nReading the configuration from args\n#####################################################\n')
            


        if self.source == 'py': # take parameters from py

            self.load_from_py(self.config_file)

    
            


        else: # needs extra evalution of parameters
            if self.source == 'json': # take parameters from json

                self.load_from_json(self.config_file)
                    
            else: # self.source == "args"

                self.load_from_args()

            
            self.interpret_params()
            
            #################### PRINT ARGUMENTS ####################
            if self.opt.verbose:
                self.print_input_arguments()

    def print_input_arguments(self):
        """
        This method returns a string representation of the input arguments passed to the script. 
        The string includes the options passed, the default value of each option, and the value of each option passed as an argument. 
        If the source is 'py', the contents of the configuration file are also included in the returned string.

        Returns:
            str: A string representation of the input arguments.
        """

        opt_dict = vars(self.opt)
        
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(opt_dict.items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        if self.source == 'py':
            message += f'\n{self.file_cont}\n'

        return message

    def get_opt(self):
        """
        This method returns the options passed to the script as an object. 

        Returns:
            object: An object containing the options passed to the script.
        """
        return self.opt

    def save_params(self, path):
        """
        This method saves the input arguments and the configuration file to a specified location as json.
        
        Args:
            path (str): The location to save the input arguments and configuration file.

        Returns:
            None: This method does not return anything.
        """

        def save_to_json(file:dict, path):
            with open(path, "w") as outfile:
                json.dump(file, outfile)
        try:
            os.makedirs(self.opt.path_to_save_pretrained_models, exist_ok=True)

            if self.source == "py":
                shutil.copy(self.config_file, os.path.join(path, os.path.join(self.config_file, os.path.basename(self.config_file))))
            elif self.source == "json":
                shutil.copy(self.config_file, os.path.join(path, os.path.join(self.config_file, os.path.basename(self.config_file))))
            save_to_json(vars(self.inp), os.path.join(path, self.opt.experiment_name + '.json'))
            
        except Exception as e:
            print("\nCould not save config file. Skipping")
            print(e)

    def parse_arguments(self):
        """
        parse_arguments method is used to parse the command line arguments for the script.

        This method sets up the argument parser using the argparse module and adds various arguments for the script.
        """


        self.parser = argparse.ArgumentParser(description="Script for training a model")
        self.parser.add_argument("-ex", "--exp_path", type=str, default='', help="Path to the experiment configuration file")

        # Logging Parameters
        self.parser.add_argument('--project_name', default='USZ_final', help='Name of the project')
        self.parser.add_argument('--verbose', '-v', choices=("True","False"), default="True", help='Turn on verbose mode')
        self.parser.add_argument('--use_wandb', choices=("True","False"), default="True", help='Use Weights & Biases for tracking')
        self.parser.add_argument('--wandbProject', type=str, default='USZ_opt', help='Name of the Weights & Biases project')
        self.parser.add_argument('--sweep', choices=("True","False"), default="True", help='Is this a sweep experiment?')
        self.parser.add_argument('--tags', nargs='+', type=str, default=['simple'], help='Tags to categorize the experiment')

        self.parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes to use')
        self.parser.add_argument('--debug', choices=("True","False"), default="False", help='Turn on debugging mode')

        # Data Parameters
        self.parser.add_argument('--dataset', default='USZ_BrainArtery_bias', help='Dataset name')
        self.parser.add_argument('--extra_dataset', default='', help='Path to extra data for training')
        self.parser.add_argument('--split_dict', default='k_fold_split2_val.json', help='Split dictionary name')
        self.parser.add_argument('--training_dataset', default='train', help='Name of the training dataset')
        self.parser.add_argument('--validation_dataset', default='val', help='Name of the validation dataset')
        self.parser.add_argument('--test_dataset', default='test', help='Name of the test dataset')
        self.parser.add_argument('--normalization', default='minmax', choices=['minmax', 'zscore'], help='Data normalization strategy')

        # Metrics and Validation
        self.parser.add_argument('--grid_validation', choices=("True","False"), default="False", help='Use grid patching for validation')
        self.parser.add_argument('--compute_mdice', choices=("True","False"), default="False", help='Compute multi-class Dice coefficient (needs more memory)')
        self.parser.add_argument('--metric_train', type=str, default="reduced", help='Metric to use for training. Options: full, reduced, none')
        self.parser.add_argument('--add_own_hausdorff', choices=("True","False"), default="False", help='Compute Hausdorff using extra memory')

        self.parser.add_argument('--shuffle_train', choices=("True","False"), default="True", help='Shuffle training set')
        self.parser.add_argument('--shuffle_validation', choices=("True","False"), default="False", help='Shuffle validation set')
        self.parser.add_argument('--shuffle_test', choices=("True","False"), default="False", help='Shuffle test set')

        # Batch Sizes and Folds
        self.parser.add_argument('--batch_size_val', type=int, default=1, help='Batch size for validation')
        self.parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
        self.parser.add_argument('--fold_id', type=int, default=0, help='ID of fold')
        self.parser.add_argument('--num_training_files', type=int, default=-1, help='Number of training files')

        self.parser.add_argument('--neighboring_vessels', choices=("True","False"), default="False", help='Include neighboring vessels')
        self.parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
        self.parser.add_argument('--collapse_classes', choices=("True","False"), default="False", help='Collapse multiple classes into one')
        self.parser.add_argument('--pretrained_weights_dict', default='', help='Path to pretrained weights dictionary')

        # Loss and Optimizers
        self.parser.add_argument('--loss', default='weightedBCE', help='Loss function. Choices: weightedBCE, mixloss, softdiceloss, focalloss')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
        self.parser.add_argument('--lr_scheduler', default='reduce', help='Learning rate scheduler')
        self.parser.add_argument('--lambda_loss', type=float, default=1.0, help='Lambda loss')
        self.parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
        self.parser.add_argument('--early_stopping', choices=("True","False"), default="False", help='Use early stopping')
        self.parser.add_argument('--patience', type=int, default=50, help='Patience for the learning rate scheduler')
        self.parser.add_argument('--number_of_epoch', type=int, default=300, help='Number of epochs')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (default: Adam)')
        self.parser.add_argument('--update_loss_epoch', type=int, default=20, help='Update loss function every n epochs (default: 20)')

        self.parser.add_argument('--positive_weight', type=float, default=100, help='Weight for positive samples (default: 100)')
        self.parser.add_argument('--negative_weight', type=float, default=1, help='Weight for negative samples (default: 1)')
        self.parser.add_argument('--target_label', type=int, default=4, help='Target label (default: 4)')

        # Preprocesing and Data Augmentation
        self.parser.add_argument('--patch_size_x', type=int, default=192, help='X patch size (default: 192)')
        self.parser.add_argument('--patch_size_y', type=int, default=192, help='Y patch size (default: 192)')
        self.parser.add_argument('--patch_size_z', type=int, default=96, help='Z patch size (default: 96)')
        self.parser.add_argument('--validate_whole_vol', choices=("True","False"), default="False", help='Validate on whole volume (default: False)')
        self.parser.add_argument('--train_whole_vol', choices=("True","False"), default="False", help='Train on whole volume (default: False)')
        self.parser.add_argument('--only_foreground', choices=("True","False"), default="True", help='Train only on foreground (default: False)')
        self.parser.add_argument('--foreground_probability', type=float, default=0.75, help='Foreground probability (default: 0.75)')
        self.parser.add_argument('--augment_probability', type=float, default=0.1, help='Augmentation probability (default: 0.1)')
        self.parser.add_argument('--patch_add', type=int, default=0, help='Add extra patches (default: 0)')
        self.parser.add_argument('--patch_xy_add', type=int, default=0, help='Add extra patches in xy direction (default: 0)')
        self.parser.add_argument('--crop_sides', choices=("True","False"), default="False", help='Crop sides (default: False)')
        self.parser.add_argument('--rand_affine', choices=("True","False"), default="True", help='Random affine transformations (default: True)')
        self.parser.add_argument('--norm_percentile', type=float, default=99.0, help='Percentile for normalization (default: 99.0)')
        self.parser.add_argument('--rand_rotate', choices=("True","False"), default="False", help='Random rotation (default: False)')
        self.parser.add_argument('--det_val_crop', choices=("True","False"), default="False", help='Determine if the validation set will be cropped deterministically or randomly')
        
        # Postprocessing
        self.parser.add_argument('--margin_crop', type=int, default=32, help='Margin for cropping the validation set')
        self.parser.add_argument('--val_threshold_cc', type=int, default=0, help='Threshold value for connected components in validation set')
        self.parser.add_argument('--val_threshold_cc_max', type=int, default=0, help='Max threshold value for connected components in validation set')
        self.parser.add_argument('--val_threshold_data', type=float, default=0.0, help='Threshold value for data in validation set')
        self.parser.add_argument('--apply_mask', choices=("True","False"), default="False", help='Determine if a mask should be applied')

        # Training Model
        self.parser.add_argument('--model_name', type=str, default='UNet', help='Name of the model')
        self.parser.add_argument('--num_blocks', type=int, default=5, help='Number of blocks in the model')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
        self.parser.add_argument('--update_layers', choices=("", "all", "decoder", "norm"), default="all", help='The layers to be updated during training. Choices: all, decoder, norm')

        self.inp = self.parser.parse_args()
        

    def cast_string_to_bool(self):
        """
        Convert string values to boolean for specified attributes in `self.opt`
        
        If an attribute is a string, it will be evaluated and set as a boolean value in `self.opt`.
        """

        for el in ["validate_whole_vol", "debug"  , "grid_validation" , "compute_mdice"  , "shuffle_train" , "shuffle_validation" , "shuffle_test" , "k_fold" , "collapse_classes" , "early_stopping" , "train_whole_vol" , "only_foreground" , "crop_sides" , "rand_affine", "rand_rotate", "neighboring_vessels", "det_val_crop", "apply_mask", "add_own_hausdorff"]:
            if hasattr(self.opt, el):
                if isinstance(vars(self.opt)[el], str):
                    vars(self.opt)[el] = eval(vars(self.opt)[el])

    def interpret_params(self):

        self.cast_string_to_bool()
        #################### GENERAL PARAMETERS ####################
        timestamp = int(time.time())
        self.opt.experiment_name = '%s_sweep_%d'%(self.opt.dataset, timestamp)
        self.opt.experiment_group_name = self.opt.experiment_name
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.opt.dataset.lower() == 'adam': # QUick fix for parameters, all case of ADAM point to the correct ADAM dataset
            self.opt.dataset = 'ADAM'

        self.opt.path_data = os.path.join(PATH_TO_DATASET, self.opt.dataset, 'data/')
        self.opt.path_split = os.path.join(PATH_TO_DATASET, self.opt.dataset, self.opt.split_dict)

        self.opt.path_to_save_pretrained_models = os.path.join(PATH_TO_PRETRAINED ,self.opt.experiment_name)

        if self.opt.extra_dataset != "":
            self.opt.path_data_extra = os.path.join(PATH_TO_DATASET, self.opt.extra_dataset, 'data/')
        else:
            self.opt.path_data_extra = None

        if self.opt.pretrained_weights_dict != "" and self.opt.pretrained_weights_dict != None :
            weights_dict = pd.read_csv(self.opt.pretrained_weights_dict)
            weights_name = weights_dict.iloc[self.opt.fold_id]['name']
            ind = weights_name[::-1].find('_')
            name = weights_name[:-ind-1]
            fold = int(weights_name[-ind:])
            assert fold == self.opt.fold_id, print(fold, self.opt.fold_id)

            self.opt.pretrained_weights = os.path.join(PATH_TO_PRETRAINED, name, str(fold), 'best_model.pth')
        else:
            self.opt.pretrained_weights = ""

        #################### LABEL PROCESSING ####################
        # Need to save this as is still used for the transformations
        old_target_label = self.opt.target_label

        if self.opt.num_classes == 1:
            label_transform = transformations.BinarizeSingleLabelTorch(label=old_target_label)
            new_target_label = 1
        elif self.opt.num_classes == 3:
            if not self.opt.neighboring_vessels:
                label_transform = transformations.ComposeTransforms([
                    transformations.CollapseLabelsTorch([1,2,3,    5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]),
                    transformations.MapLabelTorch(old_target_label,2),
                    transformations.OneHotEncodeLabel(self.opt.num_classes)
                ])
            else:
                label_transform = transformations.ComposeTransforms([
                    transformations.TakeNeighboringLabels(n_vessel_label=1, aneurysm_label=4),
                    transformations.MapLabelTorch(old_target_label,2),
                    transformations.OneHotEncodeLabel(self.opt.num_classes)
                ]) 
                
            new_target_label = 2

        elif self.opt.num_classes == 22:
            label_transform = transformations.OneHotEncodeLabel(self.opt.num_classes)
            new_target_label = old_target_label
        else:
            raise NotImplementedError(f'Num_Classes {self.opt.num_classes} is not implemented')
        
        self.opt.target_label = old_target_label

        #################### TRANSFRMATIONS ####################
        patch_size_test = (self.opt.patch_size_x, self.opt.patch_size_y, self.opt.patch_size_z) # without any changes
        if self.opt.patch_add != 0:
            self.opt.patch_size_x, self.opt.patch_size_y, self.opt.patch_size_z = int(self.opt.patch_add + self.opt.patch_xy_add+self.opt.patch_size_x), int(self.opt.patch_add+ self.opt.patch_xy_add +self.opt.patch_size_y), int(self.opt.patch_add+self.opt.patch_size_z)
            print(f"Adding {self.opt.patch_add} to patch size")

        patch_size_train = (self.opt.patch_size_x, self.opt.patch_size_y, self.opt.patch_size_z)

        # Training Set Tranformations
        self.opt.tf_train = transformations.ComposeTransforms([
                transformations.ToTensor(),  
                transformations.CropSidesThreshold() if self.opt.crop_sides else None,
                None if self.opt.train_whole_vol else transformations.CropForeground(patch_size_train, label=old_target_label) if self.opt.only_foreground else transformations.CropForegroundBackground(patch_size_train,prob_foreground=self.opt.foreground_probability, label=self.opt.target_label),
                transformations.PadToDivisible(16),
                transformations.RandAffine() if self.opt.rand_affine else None,
                transformations.RandomRotate90(prob=self.opt.augment_probability) if (patch_size_train.count(patch_size_train[0]) == len(patch_size_train) and self.opt.rand_rotate) else None,
                transformations.RandomFlip(prob=self.opt.augment_probability),
                transformations.RandElastic(sigma_range=(5,7), magnitude_range=(10,50), prob=self.opt.augment_probability),
                transformations.RandGaussianNoise(prob=0.1, std=self.opt.augment_probability),
                label_transform
                ], debug=False)

        # Validation Set Tranformations
        self.opt.tf_val = transformations.ComposeTransforms([
                transformations.ToTensor(),
                transformations.CropSidesThreshold() if self.opt.crop_sides else None,
                None if self.opt.validate_whole_vol else transformations.CropForeground(patch_size_test, label=old_target_label) if not self.opt.det_val_crop else transformations.CropForegroundCenter(target_label=old_target_label, margin = self.opt.margin_crop),
                transformations.PadToDivisible(16),
                label_transform,
                ], debug=False)

        # Test Set Tranformations
        self.opt.tf_test = transformations.ComposeTransforms([
                transformations.ToTensor(),
                transformations.PadToDivisible(16),
                label_transform
                ], debug=False)

        # Postprocessing Transformations
        self.opt.tf_post = transformations.ComposeTransforms([
            transformations_post.MaskOutSidesThreshold() if self.opt.crop_sides else None,
            transformations_post.Mask_Concentation(threshold=0.01) if self.opt.apply_mask else None,
            transformations_post.Threshold_data(self.opt.val_threshold_data) if self.opt.val_threshold_data > 0.0 else None,
            transformations_post.Threshold_cc(self.opt.val_threshold_cc, self.opt.val_threshold_cc_max) if self.opt.val_threshold_cc > 0 else None,

        ], debug=False)

        #################### LOSS FUNCTION ####################
        if self.opt.loss.lower() == 'weightedbce':
            if self.opt.num_classes == 1:
                self.opt.criterion_loss  = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.opt.positive_weight]).to(self.opt.device))
            elif self.opt.num_classes == 3: 
                self.opt.criterion_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1,self.opt.negative_weight,self.opt.positive_weight]).to(self.opt.device)) 
            elif self.opt.num_classes == 22:
                weight_list = [1] + [self.opt.negative_weight]*(self.opt.num_classes-1)
                weight_list[new_target_label] = self.opt.positive_weight
                self.opt.criterion_loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight_list).to(self.opt.device))
            else:
                raise NotImplementedError(f'Num_Classes {self.opt.num_classes} for {self.opt.loss} is not implemented')
        
        elif  self.opt.loss.lower() == 'mixloss':
            if self.opt.num_classes == 1:
                self.opt.criterion_loss = MixLoss([torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.opt.positive_weight]).to(self.opt.device)), 
                                                monai.losses.GeneralizedDiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, reduction='mean')],
                                                [0.5,0.5])
            elif self.opt.num_classes == 3: 
                self.opt.criterion_loss = MixLoss([nn.CrossEntropyLoss(weight=torch.Tensor([1,self.opt.negative_weight,self.opt.positive_weight]).to(self.opt.device)), 
                                                monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=False, sigmoid=True, reduction='mean')],
                                                [0.5,0.5])
            elif self.opt.num_classes == 22:
                weight_list = [1] + [self.opt.negative_weight]*(self.opt.num_classes-1)
                weight_list[new_target_label] = self.opt.positive_weight
                self.opt.criterion_loss = MixLoss([nn.CrossEntropyLoss(weight=torch.Tensor(weight_list).to(self.opt.device)), 
                                monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=False, sigmoid=True, reduction='mean')],
                                [0.5,0.5])
            else:
                raise NotImplementedError(f'Num_Classes {self.opt.num_classes} for {self.opt.loss} is not implemented')
        
            self.opt.criterion_loss = MixLoss([nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.opt.positive_weight]).to(self.opt.device)), DiceLoss()], [0.9,0.1])
        elif self.opt.loss.lower() == 'focalloss':
            self.opt.criterion_loss = FocalLoss(reduction = 'mean')
        elif self.opt.loss.lower() == 'diceloss':
            self.opt.criterion_loss = DiceLoss()
        elif self.opt.loss.lower() == 'focaldiceloss':
            self.opt.criterion_loss = FocalDiceLoss()
        elif self.opt.loss.lower() =='sweeploss':
            self.opt.criterion_loss = MixSweepLoss([nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.opt.positive_weight]).to(self.opt.device)), FocalDiceLoss()], [1.0,0.0])
        elif self.opt.loss.lower() == 'softdiceloss':
            if self.opt.num_classes == 1:
                self.opt.criterion_loss = monai.losses.GeneralizedDiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, reduction='mean')
            else: #self.opt.num_classes >= 1:
                self.opt.criterion_loss = monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=False, sigmoid=True, reduction='mean')
        elif self.opt.loss.lower() == 'comboloss':
            self.opt.criterion_loss = ComboLoss()
        
        else: 
            raise NotImplementedError(f'Loss {self.opt.loss} is not implemented')

        #################### VOXEL SIZE ####################

        if "111" in self.opt.dataset:
            self.opt.voxel_size = (1.0,1.0,1.0)
        elif "666" in self.opt.dataset: 
            self.opt.voxel_size = (0.6,0.6,0.6)
        else:
            self.opt.voxel_size = (0.3,0.3,0.6)
        
        #################### METRICS ####################

        if self.opt.num_classes == 1:
            self.opt.criterion_metric =  {
                    "Dice":DiceMetric(),
                    "Recall": RecallMetric(),
                    "Precision": PrecisionMetric(),
                    "Hausdorff": HausDorffMetricMonai(percentile=99),
                    "VolSimilarity": VolumetricSimilarityMetric(),
                    "HausdorffOurs": HausDorffMetric(percentile=95, voxel_size=self.opt.voxel_size) if self.opt.add_own_hausdorff else None,
                    }
        else:
            
            self.opt.criterion_metric =  {                    
                "Dice": TargetLabelMetric(DiceMetric(), new_target_label),
                "Recall": TargetLabelMetric(RecallMetric(), new_target_label),
                "Precision": TargetLabelMetric(PrecisionMetric(), new_target_label),
                "Hausdorff": TargetLabelMetric(HausDorffMetricMonai(percentile=99), new_target_label),
                "VolSimilarity": TargetLabelMetric(VolumetricSimilarityMetric(), new_target_label),
                "MDice": MultiClassDiceMetric(include_background=True) if self.opt.compute_mdice else None,
                "HausdorffOurs": TargetLabelMetric(HausDorffMetric(percentile=95, voxel_size = self.opt.voxel_size), new_target_label) if self.opt.add_own_hausdorff else None
                    }


        # Sets the training metrics
        if self.opt.metric_train.lower() == "full":
            self.opt.criterion_metric_train = self.opt.criterion_metric   
        elif self.opt.metric_train.lower() == "reduced":
            reduced_metrices = ["Dice", "Recall", "Precision"]
            self.opt.criterion_metric_train = {key:self.opt.criterion_metric[key] for key in self.opt.criterion_metric.keys() if key in reduced_metrices}
        else: #self.opt.criterion_metric_train.lower() == "none" or self.opt.criterion_metric_train.lower() == None
            self.opt.criterion_metric_train = {}



        #################### MODEL ####################
        # sets the channel size, defaults to 5 channels with sizes (16, 32, 64, 128, 256)
        channels = [int(2**(4+n)) for n in range(self.opt.num_blocks)]

        if self.opt.model_name.lower() == 'unet':
            self.opt.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.opt.num_classes,
                channels=channels,
                strides=(2, 2, 2, 2),
                kernel_size=3, 
                up_kernel_size=3, 
                num_res_units=0, 
                act='PRELU',
                norm='INSTANCE',
                dropout=self.opt.dropout,
                bias=True,
                adn_ordering='NDA'
            )  
        elif self.opt.model_name.lower() == 'attentionunet':
            self.opt.model = AttentionUnet(
                spatial_dims = 3, 
                in_channels = 1, 
                out_channels = self.opt.num_classes, 
                channels = channels, 
                strides = (2, 2, 2, 2), 
                kernel_size=3, 
                up_kernel_size=3, 
                dropout=self.opt.dropout
            )
        elif self.opt.model_name.lower() == 'segresnet':
            self.opt.model = monai.networks.nets.SegResNet(
                spatial_dims=3, 
                init_filters=8, 
                in_channels=1, 
                out_channels=self.opt.num_classes, 
                dropout_prob=self.opt.dropout, 
                act=('RELU', {'inplace': True}), 
                norm=('GROUP', {'num_groups': 8}), 
                norm_name='', num_groups=8, 
                use_conv_final=True, 
                blocks_down=(1, 2, 2, 4), 
                blocks_up=(1, 1, 1)
            )
        else: 
            raise NotImplementedError(f'Model {self.opt.model_name} is not implemented')
        
        #################### OPTIMIZER ####################

        if self.opt.update_layers == "norm":
            update_parameters = self._get_norm_parameters(self.opt.model)
        elif self.opt.update_layers == "decoder":
            update_parameters = self._get_decoder_parameters(self.opt.model)
        else:
            update_parameters = self.opt.model.parameters()

        if self.opt.optimizer.lower() == 'adamw':
            self.opt.optimizer = torch.optim.AdamW(update_parameters, lr = self.opt.learning_rate, weight_decay = self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'adam':
            self.opt.optimizer = torch.optim.Adam(update_parameters, lr = self.opt.learning_rate, weight_decay = self.opt.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.opt.optimizer} is not implemented')

    def _get_norm_parameters(self, model) -> list:
        """Returns the parameters for the normalization layers in the model.
        Args:
            model: The model to get the normalization layers from.
        Returns:
            A list of dictionaries containing the parameters for the normalization layers.
        """
        norm_params = []
        for name, param in model.named_parameters():
            if 'adn' in name:
                norm_params.append(param)
        return norm_params

    def _get_decoder_parameters(self, model) -> list:
        """Returns the parameters for the decoder layers in the model.
        Args:
            model: The model to get the decoder layers from.
        Returns:
            A list of dictionaries containing the parameters for the decoder layers.
        """
        decoder_params = []
        for mod in model.modules():
            if 'ConvTranspose3d' in type(mod).__name__:
                decoder_params.extend(list(mod.parameters()))
        return decoder_params
            
    def load_from_py(self, config_file):

        """
        Load the configuration file in Python format and store it in a Namespace object.

        Args:
            config_file (str): The file path to the configuration file.

        Returns:
            None
        """

        config_module = config_file.split('/')[-1].rstrip('.py')
        
        self.opt = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.
        self.opt = Namespace(**{key:self.opt.__dict__[key] for key in self.opt.__dict__.keys() if key[:2] != '__' and not isinstance(self.opt.__dict__[key], ModuleType) and key not in ['Tensor', 'Callable']})
        
        with open(self.config_file, 'r') as f:
            self.file_cont = f.read()

        self.replace_missing_arguments()

    def load_from_json(self, config_file):

        """
        Load the configuration file in JSON format, update it with `kwargs` if provided, and store it in a Namespace object.

        Args:
            config_file (str): The file path to the configuration file.

        Returns:
            None
        """
        
        with open(config_file, 'r') as infile:
            opt_dict = json.load(infile)
            if self.kwargs != None:
                print(f'Overwriting config file with kwargs: {self.kwargs}')
                opt_dict.update(self.kwargs)
                
            self.opt = Namespace(**opt_dict)
        self.file_cont = str(self.opt)

        self.replace_missing_arguments()

    def replace_missing_arguments(self):
        """
        Replace missing arguments with default values.
        """

        if not hasattr(self.opt, 'split_dict'):
            self.opt.split_dict = 'k_fold_split2_val.json'
        if not hasattr(self.opt, 'rand_rotate'):
            self.opt.rand_rotate = False
        if not hasattr(self.opt, 'neighboring_vessels'):
            self.opt.neighboring_vessels = False
        if not hasattr(self.opt, 'det_val_crop'):
            self.opt.det_val_crop = False
        if not hasattr(self.opt, 'val_threshold_data'):
            self.opt.val_threshold_data = 1.0
        if not hasattr(self.opt, 'val_threshold_cc'):
            self.opt.val_threshold_cc = 100
        if not hasattr(self.opt, 'pretrained_weights_dict'):
            self.opt.pretrained_weights_dict = ""
        if not hasattr(self.opt, 'apply_mask'):
            self.opt.apply_mask = False
        if not hasattr(self.opt, 'val_threshold_cc_max'):
            self.opt.val_threshold_cc_max = -1
        if not hasattr(self.opt, 'add_own_hausdorff'):
            self.opt.add_own_hausdorff = False
        if not hasattr(self.opt, 'normalization'):
            self.opt.normalization = 'minmax'
        if not hasattr(self.opt, 'rand_affine'):
            self.opt.rand_affine = False
        if not hasattr(self.opt, 'grid_validation'):
            self.opt.grid_validation = False
        if not hasattr(self.opt, 'num_blocks'):
            self.opt.num_blocks = 5
        if not hasattr(self.opt, 'metric_train'):
            self.opt.metric_train = "reduced" 
            
    def load_from_args(self):
        """
        Load the configuration from command-line arguments and store it in a Namespace object.

        Returns:
            None
        """

        self.opt = copy.deepcopy(self.inp)