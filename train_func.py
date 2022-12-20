from train.utils import *
from train.metrics import MetricesStruct
from train.callbacks import *
from train.logging import *

from itertools import count
import matplotlib
import torch
import torch.optim as optim
import numpy as np
import pdb
import copy
from tqdm import tqdm
import torchio as tio
import GPUtil
matplotlib.use("Agg")
from torch.utils.tensorboard import SummaryWriter
# ===============
# Function for training segmentation network
# ===============


def _train(exp_config, model, loader_train, epoch, device, optimizer, criterion_loss):
     # ==================================
    # TRAIN
    # ==================================
    model.train() # Switch on training mode            
    running_loss_train_total  = 0.0

    train_metrics = MetricesStruct(exp_config.criterion_metric_train, prefix='train_')
    counter = 0
    print('Training epoch ', str(epoch))
    with tqdm(loader_train, unit='batch') as tepoch:
        #timer.take_time('Load Data')
        for data, target, norm_params, name in tepoch:
            #timer.take_time('Feed through network')
            data, target, norm_params = data.to(device).float(), target.to(device).float(), norm_params.to(device)
            optimizer.zero_grad()
            pred_logits = model(data)
            #timer.take_time('Compute Loss')
            loss_train = criterion_loss(pred_logits, target)
            
            if exp_config.num_classes > 1:
                pred = torch.softmax(pred_logits, dim=1)
            else:
                pred = torch.sigmoid(pred_logits)
            del pred_logits #for memory saving

            train_metrics.update(pred, target)
            #timer.take_time('Update Network')
            loss_train.backward()
            optimizer.step()

            running_loss_train_total += loss_train.item()
            
            counter += 1

            epoch_log_train = {"loss":loss_train.item()}
            epoch_log_train.update(train_metrics.get_last_scores())
            tepoch.set_postfix(epoch_log_train)
            #timer.take_time('Load Data')

    
    running_loss_train_total = running_loss_train_total / counter

    print('Finished training epoch')
    return running_loss_train_total, train_metrics


def _validate(exp_config, model, loader_val, epoch, device, val_loss, logger):
    # =========
    # VAL
    # =========
    #timer = Timer()
    #print("Initialization")
    #timer.take_time('Inference')
    #GPUtil.showUtilization()
    model.eval() # Switch on evaluation mode
    with torch.no_grad():
        running_loss_val_total = 0.0
        val_metrics = MetricesStruct(exp_config.criterion_metric, prefix='val_')
        counter = 0
        
        

        with tqdm(loader_val, unit='batch') as tepoch:
            for data, target, norm_params, name in tepoch:
                data, target, norm_params = data.float().to(device), target.float().to(device), norm_params

                if exp_config.grid_validation:
                    
                    patch_subject = tio.Subject(data= tio.ScalarImage(tensor=data.squeeze(0)), target = tio.LabelMap(tensor=target.squeeze(0)))
                    grid_sampler = tio.inference.GridSampler(subject = patch_subject,patch_size = (exp_config.patch_size_x, exp_config.patch_size_y, exp_config.patch_size_z), patch_overlap = 0)
                    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode ="crop")
                    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=exp_config.batch_size_val)
                    patch_loss = 0.0
                    for patches_batch in patch_loader:
                        data_patch = patches_batch['data'][tio.DATA].to(device)
                        locations_patch = patches_batch[tio.LOCATION]
                        batch_pred_logits = model(data_patch)
                        patch_loss += val_loss(batch_pred_logits, patches_batch['target'][tio.DATA].to(device))
                        if exp_config.num_classes > 1:
                            pred = torch.softmax(batch_pred_logits, dim=1)
                        else:
                            pred = torch.sigmoid(batch_pred_logits)

                        aggregator.add_batch(pred, locations_patch)

                    pred = aggregator.get_output_tensor().unsqueeze(0).to(device)
                    loss_val = patch_loss/len(patch_loader)
                else:

                    pred_logits = model(data)
                    loss_val = val_loss(pred_logits.to(device), target.to(device))

                    if exp_config.num_classes > 1:
                        pred = torch.softmax(pred_logits, dim=1)
                    else:
                        pred = torch.sigmoid(pred_logits)



                if hasattr(exp_config, "tf_post"):
                    sample = {'image': data.squeeze(0), 'target': pred.squeeze(0)}
                    sample = exp_config.tf_post(sample)
                    pred = sample['target'].unsqueeze(0)

                val_metrics.update(pred, target)

                running_loss_val_total += loss_val.item()
                
                counter += 1
                epoch_log_val = {"val_loss":loss_val.item()}
                epoch_log_val.update(val_metrics.get_last_scores())
                tepoch.set_postfix(epoch_log_val)

                logger.visualize_val(data, target, pred, name, epoch, num=1)
                #GPUtil.showUtilization()
                #timer.take_time('done')
        running_loss_val_total = running_loss_val_total / counter

        print('Finished validation epoch')

        return running_loss_val_total, val_metrics

def train_segmentation_network(
        exp_config,
        model, 
        optimizer, 
        loader_train = None, 
        loader_val = None, 
        loader_test = None, 
        path_to_save_pretrained_models = '', 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        exp_cont = '',
        exp_name = None):

    # ==================================
    # INITIALIZATION
    # ==================================

    #timer = Timer()
    #timer.take_time('Initialization')

    logger = TrainLogger(exp_config, 
                            model, 
                            exp_name, 
                            path_to_save_pretrained_models, 
                            wandb_log=True, 
                            tb_log = False, 
                            save_images=False, 
                            tb_visualize=False, 
                            wandb_visualize=True,
                            debug = exp_config.debug)

    model_saver = ModelSaver(path_to_save_pretrained_models, 
                            save_every_n_epoch=10)

    print(exp_cont)

    if exp_config.lr_scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=int(exp_config.patience/2), threshold=1e-5, verbose=True)
    elif exp_config.lr_scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(loader_train), epochs=exp_config.number_of_epoch, verbose=True)
    elif exp_config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(exp_config.patience * 0.8), gamma=0.1, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda x:exp_config.lambda_loss, verbose=True)
    if exp_config.early_stopping:
        early_stopping = EarlyStopping(patience=exp_config.patience, min_delta=0)

    criterion_loss = exp_config.criterion_loss
    val_loss = copy.deepcopy(criterion_loss).to(device)
    
    if exp_config.pretrained_weights != '':
        state_dict = torch.load(exp_config.pretrained_weights)
        model.load_state_dict(state_dict)
    
    model.to(device)
    #timer.take_time('Start Training')
    for epoch in range(exp_config.number_of_epoch):
       
        running_loss_train_total, train_metrics = _train(exp_config=exp_config, model=model, loader_train=loader_train, epoch=epoch, device=device, optimizer=optimizer, criterion_loss=criterion_loss)

        running_loss_val_total, val_metrics = _validate(exp_config=exp_config, model=model, loader_val=loader_val, epoch=epoch, device=device, val_loss=val_loss, logger = logger)
        
        model_saver.update(running_loss_val_total, epoch, model)

        log_dict = {"training_loss": running_loss_train_total,
                    "validation_loss": running_loss_val_total,
                    "learning_rate": get_current_lr(optimizer)
                    }
        log_dict.update(val_metrics.get_scores())
        log_dict.update(train_metrics.get_scores())
        logger.log(log_dict=log_dict, epoch = epoch, verbose=True)

        if exp_config.lr_scheduler in  ['reduce', 'cycle']:
            scheduler.step(running_loss_val_total)
        else:
            scheduler.step()
        
        if exp_config.early_stopping:
            early_stopping(running_loss_val_total)
            if early_stopping.early_stop:
                model_saver.save_final_model(model)
                break

        if exp_config.loss.lower() == "sweeploss" and epoch%exp_config.update_loss_epoch==0 and epoch != 0:
            criterion_loss.update_ratio(min(1.0, epoch/exp_config.update_loss_epoch/5))
            print(f"Updating Loss: {criterion_loss.get_weights()}")
            
        logger.end_of_epoch()


    model_saver.save_final_model(model)
    logger.end_of_training
    return logger.get_score_logger()


