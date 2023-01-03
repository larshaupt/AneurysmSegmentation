import argparse
from pathlib import Path
import os
from data import transformations, transformations_post
import numpy as np
import time
import torch
from train.metrics import *
from train.losses import *
from train.utils import *
import monai
from monai.networks.nets import UNet, AttentionUnet, UNETR, AHNet
import json
from importlib.machinery import SourceFileLoader
from argparse import Namespace
from types import ModuleType
import shutil
import copy
import pdb
import torchio as tio

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

            
            if not hasattr(self.opt, 'criterion_metric_train'):
                self.opt.criterion_metric_train = self.opt.criterion_metric
            


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
        return self.opt

    def save_params(self, path):
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

        self.parser = argparse.ArgumentParser(description="Script for training")
        self.parser.add_argument("-ex", "--exp_path", type=str, default='', help="Path to experiment config file")

        self.parser.add_argument('--project_name', default='USZ_opt', help='project name')
        self.parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode?')
        self.parser.add_argument('--use_wandb', action='store_true', help='use wandb?')
        self.parser.add_argument('--wandbProject', type=str, default='USZ_opt', help='name of the wandb project to use')
        self.parser.add_argument('--sweep', action='store_true', help='is this a sweep?')
        self.parser.add_argument('--tags', nargs='+', type=str, default=['simple'])

        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
        self.parser.add_argument('--debug',choices=('True','False'), default='False', help='Debugging')

        self.parser.add_argument('--dataset', default='USZ_hdf5d2', help='dataset name')
        self.parser.add_argument('--path_data', default='', help='path_data')
        self.parser.add_argument('--extra_dataset', default='', help='path_data extra for training')
        self.parser.add_argument('--path_split', default='', help='path_split')
        self.parser.add_argument('--split_dict', default='k_fold_split2_val.json', help='split_dict')
        self.parser.add_argument('--path_to_save_pretrained_models', default='/srv/beegfs02/scratch/brain_artery/data/training/pre_trained', help='path_to_save_pretrained_models')
        self.parser.add_argument('--training_dataset', default='train', help='dataset name')
        self.parser.add_argument('--validation_dataset', default='val', help='dataset name')
        self.parser.add_argument('--test_dataset', default='test', help='dataset name')
        self.parser.add_argument('--grid_validation', choices=('True','False'), default='False', help='Use grid patching for validation?')
        self.parser.add_argument('--compute_mdice', choices=('True','False'), default='False', help='Compute Multiclass Dice? Needs more memory')
        self.parser.add_argument('--metric_train', type=str, default="reduced", help='Options are full, reduced, none')


        self.parser.add_argument('--shuffle_train',choices=('True','False'), default='True', help='shuffle_train')
        self.parser.add_argument('--shuffle_validation',choices=('True','False'), default='False', help='shuffle_validation')
        self.parser.add_argument('--shuffle_test',choices=('True','False'), default='False', help='shuffle_test')

        self.parser.add_argument('--batch_size_val', type=int, default=1, help='val. input batch size')
        self.parser.add_argument('--batch_size', type=int, default=2, help=' input batch size')
        self.parser.add_argument('--fold_id', type=int, default=0, help='fold id')
        self.parser.add_argument('--num_training_files', type=int, default=-1, help='num_training_files')
        self.parser.add_argument('--k_fold',choices=('True','False'), default='False', help='Doing k-fold?')
        self.parser.add_argument('--k_fold_k', type=int, default = 5)
        

        self.parser.add_argument('--neighboring_vessels', choices=('True','False'), default='False', help='neighboring_vessels')
        self.parser.add_argument('--num_classes', type=int, default=1, help=' number of classes')
        self.parser.add_argument('--collapse_classes', choices=('True','False'), default='False', help='Wether to collapse multiple classes into one')
        self.parser.add_argument('--pretrained_weights', default='', help='path to pretrained_weights')

        self.parser.add_argument('--loss', default='weightedBCE', help='loss')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help=' initial learning rate')
        self.parser.add_argument('--lr_scheduler', default='reduce', help='Which learning rate scheduler')
        self.parser.add_argument('--lambda_loss', type=float, default=1.0, help=' lambda loss')
        self.parser.add_argument('--weight_decay', type=float, default=0.0, help=' lambda loss')
        self.parser.add_argument('--early_stopping',choices=('True','False'), default='False', help='Doing early stopping')
        self.parser.add_argument('--patience', type=int, default=50, help=' patience')
        self.parser.add_argument('--number_of_epoch', type=int, default=300, help=' number of epochs')
        self.parser.add_argument('--optimizer', type=str, default='adam', help=' Optimizer')
        self.parser.add_argument('--update_loss_epoch', type=int, default='20', help=' Update Loss function every n epochs')

        
        self.parser.add_argument('--positive_weight', type=float, default=100, help=' positive_weight')
        self.parser.add_argument('--negative_weight', type=float, default=1, help=' negative_weight')
        self.parser.add_argument('--target_label', type=int, default=4, help=' target_label')

        
        self.parser.add_argument('--patch_size_x', type=int, default=192, help=' x patch size')
        self.parser.add_argument('--patch_size_y', type=int, default=192, help=' y patch size')
        self.parser.add_argument('--patch_size_z', type=int, default=96, help=' z patch size')
        self.parser.add_argument('--validate_whole_vol', choices=('True','False'), default='False')
        self.parser.add_argument('--train_whole_vol', choices=('True','False'), default='False')
        self.parser.add_argument('--only_foreground',choices=('True','False'), default='False')
        self.parser.add_argument('--foreground_probability', type=float, default=0.75)
        self.parser.add_argument('--extra_cropping',choices=('True','False'), default='False')
        self.parser.add_argument('--augment_probability', type=float, default=0.1)
        self.parser.add_argument('--patch_add', type=int, default=0)
        self.parser.add_argument('--patch_xy_add', type=int, default=0)
        self.parser.add_argument('--crop_sides',choices=('True','False'), default='False')
        self.parser.add_argument('--rand_affine',choices=('True','False'), default='True')
        self.parser.add_argument('--norm_percentile', type=float, default=99.0)
        self.parser.add_argument('--rand_rotate',choices=('True','False'), default='False')

        self.parser.add_argument('--model_name', type=str, default='UNet', help='name of model')
        self.parser.add_argument('--num_blocks', type=int, default=5)
        self.parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')


        self.inp = self.parser.parse_args()
        

    def interpret_params(self):

        for el in ["validate_whole_vol", "debug"  , "grid_validation" , "compute_mdice"  , "shuffle_train" , "shuffle_validation" , "shuffle_test" , "k_fold" , "collapse_classes" , "early_stopping" , "train_whole_vol" , "only_foreground" , "extra_cropping" , "crop_sides" , "rand_affine", "rand_rotate", "neighboring_vessels"]:
            if hasattr(self.opt, el):
                if isinstance(vars(self.opt)[el], str):
                    vars(self.opt)[el] = eval(vars(self.opt)[el])
        #################### GENERAL PARAMETERS ####################
        timestamp = int(time.time())
        self.opt.experiment_name = '%s_sweep_%d'%(self.opt.dataset, timestamp)
        self.opt.experiment_group_name = self.opt.experiment_name
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.opt.dataset.lower() == 'adam': # QUick fix for parameters, all case of ADAM point to the correct ADAM dataset
            self.opt.dataset = 'ADAM'
        if self.opt.path_data == "":
            self.opt.path_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/data/'%(self.opt.dataset)
        if self.opt.path_split == "":
            self.opt.path_split = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/%s'%(self.opt.dataset, self.opt.split_dict)
            print(self.opt.path_split)
        self.opt.path_to_save_pretrained_models = os.path.join(self.opt.path_to_save_pretrained_models ,self.opt.experiment_name)

        if self.opt.extra_dataset != "":
            self.opt.path_data_extra = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/%s/data/'%(self.opt.extra_dataset)
        else:
            self.opt.path_data_extra = None
        if not hasattr(self.opt, 'rand_affine'):
            self.opt.rand_affine = False
        if not hasattr(self.opt, 'grid_validation'):
            self.opt.grid_validation = False
        if self.opt.sweep:
            self.opt.name = self.opt.name + '_' + os.environ['SLURM_ARRAY_TASK_ID']

        #################### LABEL PROCESSING ####################
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

        self.opt.tf_train = transformations.ComposeTransforms([
                transformations.ToTensor(),  
                transformations.CropSidesThreshold() if self.opt.crop_sides else None,
                None if self.opt.train_whole_vol else transformations.CropForeground(patch_size_train, label=old_target_label) if self.opt.only_foreground else transformations.CropForegroundBackground(patch_size_train,prob_foreground=self.opt.foreground_probability, label=self.opt.target_label),
                transformations.CropRandom((np.array(patch_size_train)/2).astype(int), prob= self.opt.augment_probability) if self.opt.extra_cropping else None,
                transformations.PadToDivisible(16),
                transformations.RandAffine() if self.opt.rand_affine else None,
                transformations.RandomRotate90(prob=self.opt.augment_probability) if (patch_size_train.count(patch_size_train[0]) == len(patch_size_train) and self.opt.rand_rotate) else None,
                transformations.RandomFlip(prob=self.opt.augment_probability),
                transformations.RandElastic(sigma_range=(5,7), magnitude_range=(10,50), prob=self.opt.augment_probability),
                transformations.RandGaussianNoise(prob=0.1, std=self.opt.augment_probability),
                label_transform
                ], debug=False)

        self.opt.tf_val = transformations.ComposeTransforms([
                transformations.ToTensor(),
                transformations.CropSidesThreshold() if self.opt.crop_sides else None,
                transformations.CropForeground(patch_size_test, label=old_target_label) if not self.opt.validate_whole_vol else None,
                transformations.PadToDivisible(16),
                label_transform,
                ], debug=False)

        self.opt.tf_test = transformations.ComposeTransforms([
                transformations.ToTensor(),
                transformations.PadToDivisible(16),
                label_transform
                ], debug=False)

        self.opt.tf_post = transformations.ComposeTransforms([
            transformations_post.MaskOutSidesThreshold() if self.opt.crop_sides else None
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
                self.opt.criterion_loss = MixLoss(torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.opt.positive_weight]).to(self.opt.device)), 
                                                monai.losses.GeneralizedDiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, reduction='mean'),
                                                [0.5,0.5])
            elif self.opt.num_classes == 3: 
                self.opt.criterion_loss = MixLoss(nn.CrossEntropyLoss(weight=torch.Tensor([1,self.opt.negative_weight,self.opt.positive_weight]).to(self.opt.device)), 
                                                monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=False, sigmoid=True, reduction='mean'),
                                                [0.5,0.5])
            elif self.opt.num_classes == 22:
                weight_list = [1] + [self.opt.negative_weight]*(self.opt.num_classes-1)
                weight_list[new_target_label] = self.opt.positive_weight
                self.opt.criterion_loss = MixLoss(nn.CrossEntropyLoss(weight=torch.Tensor(weight_list).to(self.opt.device)), 
                                monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=False, sigmoid=True, reduction='mean'),
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
        else: 
            raise NotImplementedError(f'Loss {self.opt.loss} is not implemented')

        #################### METRICS ####################
        if self.opt.num_classes == 1:
            self.opt.criterion_metric =  {
                    "Dice":DiceMetric(),
                    "Recall": RecallMetric(),
                    "Precision": PrecisionMetric(),
                    "Hausdorff": HausDorffMetricMonai(percentile=99),
                    "VolSimilarity": VolumetricSimilarityMetric()
                    }
        else:
            self.opt.criterion_metric =  {                    
                "Dice": TargetLabelMetric(DiceMetric(), new_target_label),
                "Recall": TargetLabelMetric(RecallMetric(), new_target_label),
                "Precision": TargetLabelMetric(PrecisionMetric(), new_target_label),
                "MDice": MultiClassDiceMetric(include_background=True) if self.opt.compute_mdice else None
                    }

        if not hasattr(self.opt, 'metric_train'):
            self.opt.criterion_metric_train = self.opt.criterion_metric      
        elif self.opt.metric_train.lower() == "full":
            self.opt.criterion_metric_train = self.opt.criterion_metric   
        elif self.opt.metric_train.lower() == "reduced":
            reduced_metrices = ["Dice", "Recall", "Precision"]
            self.opt.criterion_metric_train = {key:self.opt.criterion_metric[key] for key in self.opt.criterion_metric.keys() if key in reduced_metrices}
        else: #self.opt.criterion_metric_train.lower() == "none" or self.opt.criterion_metric_train.lower() == None
            self.opt.criterion_metric_train = {}





        #################### MODEL ####################
        if hasattr(self.opt, 'num_blocks'):
            channels = [int(2**(4+n)) for n in range(self.opt.num_blocks)]
        else:
            channels = (16, 32, 64, 128, 256)

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

        if self.opt.optimizer.lower() == 'adamw':
            self.opt.optimizer = torch.optim.AdamW(self.opt.model.parameters(), lr = self.opt.learning_rate, weight_decay = self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'adam':
            self.opt.optimizer = torch.optim.Adam(self.opt.model.parameters(), lr = self.opt.learning_rate, weight_decay = self.opt.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.opt.optimizer} is not implemented')


            

    def load_from_py(self, config_file):
        config_module = config_file.split('/')[-1].rstrip('.py')
        
        self.opt = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.
        self.opt = Namespace(**{key:self.opt.__dict__[key] for key in self.opt.__dict__.keys() if key[:2] != '__' and not isinstance(self.opt.__dict__[key], ModuleType) and key not in ['Tensor', 'Callable']})
        
        with open(self.config_file, 'r') as f:
            self.file_cont = f.read()

    def load_from_json(self, config_file):
        
        with open(config_file, 'r') as infile:
            opt_dict = json.load(infile)
            if self.kwargs != None:
                opt_dict.update(self.kwargs)
            self.opt = Namespace(**opt_dict)
        self.file_cont = str(self.opt)

        if not hasattr(self.opt, 'split_dict'):
            self.opt.split_dict = 'k_fold_split2_val.json'

        if not hasattr(self.opt, 'rand_rotate'):
            self.opt.rand_rotate = False

        if not hasattr(self.opt, 'neighboring_vessels'):
            self.opt.neighboring_vessels = False

    def load_from_args(self):
        self.opt = copy.deepcopy(self.inp)