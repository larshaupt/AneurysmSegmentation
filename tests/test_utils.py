import sys
sys.path.append('/srv/beegfs02/scratch/brain_artery/data/Segmentation3D')

import train.utils as ut
from data import  HDF5Dataset3D

from preprocessing.preprocessing_utils import animate_img, save_to_nifti
from train.metrics import MetricesStruct

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np


def make_predictions(
            experiment_names,
            pre_trained_path,
            epoch = 'best_model.pth',
            save_path = None, 
            img = True,  
            gifs = False, 
            nifti = False, 
            split = 'test', 
            num = -1,
            num_slices = 5, 
            tf_mode='val',
            voxel_size = (1.0,1.0,1.0),
            save = True,
            binarize_target = True,
            config_overwrite:dict = None,
            postprocessing:bool = False,
            postfix = "",
            aggregation_strategy = 'majority',
            factor = 0.8,
            softmax=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = ut.Predictor(experiment_names, pre_trained_path, epoch=epoch, config_overwrite = config_overwrite, postprocessing=False, device=device)
    exp_config = predictor.get_exp_config(experiment_names[0])
    data_names = ut.load_split(exp_config.path_split.replace("_downscaled", "d"), exp_config.fold_id, split=split)
    for el in data_names:
        el['mask'] = el['target'].replace("_y", "_mask")

    if tf_mode == 'train':
        tf = exp_config.tf_train
    elif tf_mode == 'val' and hasattr(exp_config, 'tf_val'):
        tf = exp_config.tf_val 
    else:
        tf = exp_config.tf_test
    path_to_gifs, path_to_imgs, path_to_nifti = None, None, None
    ds_data = HDF5Dataset3D.HDF5Dataset3D(exp_config, exp_config.path_data.replace("_downscaled", "d"), data_names, tf, mask_path=exp_config.path_data)
    data_loader = torch.utils.data.DataLoader(ds_data, batch_size = 1, shuffle = False, num_workers = 0, pin_memory=False)
    data_generator = iter(data_loader)
    if save:
        save_path = os.path.join(save_path, predictor.get_all_exp_names_comp() + postfix)
        model_pred_path_split = os.path.join(save_path, f'{split}_tf{tf_mode}')
        if gifs:
            path_to_gifs = os.path.join(model_pred_path_split, 'gifs/')
        if img:
            path_to_imgs = os.path.join(model_pred_path_split, 'img/')
        if nifti:
            path_to_nifti = os.path.join(model_pred_path_split, 'nifti/')
        model_scores_path = os.path.join(model_pred_path_split, "scores.csv")

        ut.make_dirs([save_path, model_pred_path_split, path_to_gifs, path_to_imgs, path_to_nifti])

    metric = MetricesStruct(exp_config.criterion_metric, prefix='')
    #next(data_generator)

    for i, (data, target, mask ,norm_params, name) in enumerate(data_generator):
        
        print(f"Predicting {name}")
        if i >= num and num!=-1:
            break
        
        pred = predictor.predict_all_combine(data, aggregation_strategy=aggregation_strategy, factor = factor, softmax=softmax)

        if postprocessing:
            
            sample = {'image': data.squeeze(0), 'pred': pred.squeeze(0), 'mask': mask.squeeze(0)}
            sample = exp_config.tf_post(sample)
            pred = sample['pred'].unsqueeze(0)

        metric.update(pred, target, el_name=name[0])
        print(metric.get_last_scores())
        if save:
            save_pred(data, target, 
                norm_params, 
                name, 
                pred ,
                num_slices = num_slices, 
                path_to_gifs=path_to_gifs, 
                path_to_imgs=path_to_imgs, 
                path_to_nifti = path_to_nifti, 
                voxel_size=voxel_size, 
                target_label=exp_config.target_label)

    scores = metric.get_single_score_per_name()
    print(f"Scores for {predictor.get_all_exp_names()} with {tf_mode} transform on {split} split:")
    scores_df = pd.DataFrame(data=scores).transpose()
    scores_df.name = predictor.get_all_exp_names_comp()
    print(scores_df.to_string())
    if save:
        scores_df.to_csv(model_scores_path)
    scores_df.to_csv(predictor.get_all_exp_names_comp() + postfix + ".csv")

    return scores_df




def save_pred(
        data, 
        target, 
        norm_params, 
        name, pred,
        num_slices = 10,  
        path_to_gifs=None, 
        path_to_imgs=None, 
        path_to_nifti = None,
        voxel_size = (0.3,0.3,0.6),
        binarize_target = True,
        target_label = 4):

    print(f'Saving {name} to {path_to_imgs} ')


    target_channel = target_label if pred.shape[1] >=target_label else pred.shape[1] - 1
    pred_bin = ut.binarize(pred).detach().numpy()[0,0,...]
    if binarize_target:
        target = ut.binarize(target)
        target = target.detach().numpy()[0,0,...]
    else:
        target = target.detach().numpy()[0,target_channel,...]
    
    pred = pred.detach().numpy()[0,target_channel,...]


    data = data.detach().numpy()[0,0,...]

    name = name[0]

    norm_params = norm_params.detach().numpy()
    min_value, perc99_value = norm_params[0,0], norm_params[0,1]
    data = data*(perc99_value-min_value) + min_value
    save_dict = {
        'pred_bin': pred_bin,
        'pred': pred,
        'target' : target,
        'data' :  data
    }

    if path_to_gifs:
        path_to_single_gif = os.path.join(path_to_gifs, str(name))
        ut.make_dirs([path_to_single_gif])
    
        for key in save_dict:
            item = np.squeeze(save_dict[key])
                
            gif_path = os.path.join(path_to_single_gif, key + '.gif')
            animate_img(item, save_path=gif_path, size='large')

    if path_to_imgs:

        path_to_single_img = os.path.join(path_to_imgs, str(name))
        ut.make_dirs([path_to_single_img])

        for key in save_dict:
            item = np.squeeze(save_dict[key])
            slices = ut.find_slices(target, num=num_slices, ratio_pos=0.75)
            for count_slice, slice in enumerate(slices):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                img_path = os.path.join(path_to_single_img, str(count_slice) + '_' + key + '.png')
                ax.imshow(item[:,:,int(slice[-1])])
                ax.set_axis_off()
                fig.savefig(img_path)
                plt.close(fig)

    if path_to_nifti:
        path_to_single_nifti = os.path.join(path_to_nifti, str(name))
        ut.make_dirs([path_to_single_nifti])
        
        for key in save_dict:
            item = np.squeeze(save_dict[key])
            nifti_path = os.path.join(path_to_single_nifti, key + '.nii.gz')
            try:
                save_to_nifti(item, nifti_path, voxel_size)
            except:
                print(f"Could not save {nifti_path} to nifti")
                