# %%
import sys
sys.path.append('../')

from data import transformations
from data import data_loader, HDF5Dataset3D
from train.utils import *
from train.metrics import MetricesStruct
from utils import *

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cc3d
from torchmetrics import Dice, Precision, Recall, AUROC
from monai.metrics import DiceMetric
from scipy.stats import multivariate_normal
from nilearn.plotting import plot_anat, plot_roi

path_to_pretrained_weights = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained'


voxel_size = (0.3,0.3,0.6)
model_name = 'best_model.pth'
# %%

model_names = ['USZ_BrainArtery_bias_sweep_1672373421']
models = []
exp_configs = []
for experiment_name in model_names:
    config_file = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/%s/%s.json' %(experiment_name, experiment_name) 
    exp_config = load_opt(config_file)
    fold_id = exp_config.fold_id      
    model_weight_path = '/srv/beegfs02/scratch/brain_artery/data/training/pre_trained/%s/%s/%s' %(experiment_name, fold_id,model_name)   
    
    model = load_model(exp_config, model_weight_path)
    models.append(model)
    exp_configs.append(exp_config)

# %%
exp_config = exp_configs[0]
split = 'test'
data_names = load_split(exp_config.path_split.replace("_downscaled", "d"), exp_config.fold_id, split=split)
tf_mode = 'test'

data_generator = load_data_generator(exp_config, data_names, tf_mode=tf_mode, batch_size=1)

#%%

data, target, mask,norm_params, name = next(data_generator)

# %%
pred = torch.zeros_like(data, dtype=torch.float32)

for model in models:
    pred_m = inference(data, model, onehot=False)
    if pred_m.shape[1] == 3:
        pred_m = pred_m[:,2,...].unsqueeze(1)
    elif pred_m.shape[1] == 22:
        pred_m = pred_m[:,4,...].unsqueeze(1)
    pred += pred_m

pred = pred/len(models)

# %%
pred = binarize(pred, 0.5)
#pred_mask = scipy.ndimage.binary_opening(pred>0, iterations=1)
#pred_mask = scipy.ndimage.binary_closing(pred>0, iterations=2)

# %%

#pred = take_n_largest_components(pred, n=15)
pred = remove_small_components(pred, thr=150)
pred = threshold_predictions(pred,data, thr=1.0)
pred = remove_n_largest_components(pred, n=1)
pred = clip_outside_values(pred, thr=10.0)


#%%
pred_t = torch.Tensor(pred).view(1,*pred.shape[-4:])
metric = MetricesStruct(exp_config.criterion_metric, prefix='')
#pred_t = torch.nn.functional.one_hot(pred_t, num_classes=3).view(1,-1, *pred_t.shape[-3:])
metric.update(pred_t, target, el_name=name[0])
scores = metric.get_single_score_per_name()
metric.print()
# %%
pred = binarize(pred)
target = binarize(target)
pred = pred.detach().numpy()[0,0,...]
target = target.detach().numpy()[0,0,...]
data = data.detach().numpy()[0,0,...]
#%%
import cc3d

labels_pred, N_pred = cc3d.largest_k(
  pred, k=10, 
  connectivity=26, delta=0,
  return_N=True,
)
labels_target, N_target = cc3d.connected_components(target, connectivity=26, return_N=True)

pred_stat = cc3d.statistics(labels_pred)
target_stat = cc3d.statistics(labels_target)

print_pretty = lambda x: pd.DataFrame({key:[str(el) for el in x[key]] for key in x.keys()})

# %%


def normalize_loc(loc, shape):
    loc_new = loc.copy()
    for i in range(len(loc_new)):
        loc_new[i] = float(loc_new[i])/shape[i]
    return loc_new
normalize_loc(pred_stat['centroids'][0], target.shape)
# %%

for label, image in cc3d.each(labels_target, binary=False, in_place=True):
    z_slices = np.nonzero(image)
    z_slice = z_slices[-1][int(len(z_slices[2])/2)]
    plt.imshow(image[:,:,z_slice]) # stand in for whatever you'd like to do

#%%
from scipy.stats import multivariate_normal
cov = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_cov.npy')
mean = np.load('/scratch/lhauptmann/segmentation_3D/data_analysis/location_gaussian_mean.npy')
shape = target.shape
loc_prob = multivariate_normal(mean=mean, cov=cov)
for comp in target_stat['centroids']:
    print(comp,normalize_loc(comp, shape),  loc_prob.pdf(normalize_loc(comp, shape)))


#%%
num = -1
for i, (data, target, mask,norm_params, name) in enumerate(data_generator):
    if i >= num and num!=-1:
        break
    print(name)
    pred1 = model(data)
    pred1 = torch.sigmoid(pred1)
    pred2 = model2(data)
    pred12 = torch.sigmoid(pred2)
    pred = (pred1+pred2)/2

    pred= binarize(pred)
    target = binarize(target)
    #pred = take_n_largest_components(pred, n=15)
    #pred = clip_outside_values(pred, thr=0.1)
    metric.update(pred, target, el_name=name[0])


scores = metric.get_single_score_per_name()
print(f"Scores for {experiment_name} with {tf_mode} transform on {split} split:")
scores_df = pd.DataFrame(data=scores).transpose()
print(scores_df)

# %%


# %%
a = clip_outside_values(pred, thr =1)
# %%

metric_w = MetricesStruct(exp_config.criterion_metric, prefix='')
metric_wo = MetricesStruct(exp_config.criterion_metric, prefix='')
for data, target, norm_params, name in data_loader:


    pred = model(data)

    if exp_config.num_classes > 1:
        pred = torch.softmax(pred, dim=1)
    else:
        pred = torch.sigmoid(pred)
    pred = binarize(pred, 0.5)


    metric_wo.update(binarize(pred), target, el_name=name[0])

    pred = take_n_largest_components(pred, n=10)
    pred = clip_outside_values(pred, thr=1)


    metric_w.update(binarize(pred), target, el_name=name[0])


# %%
scores = metric_wo.get_single_score_per_name()
print(f"Scores for {experiment_name} with {tf_mode} transform on {split} split:")
scores_df = pd.DataFrame(data=scores).transpose()
print(scores_df)
# %%
pre_dict = {0: 'Adam_sweep_1672475897_0',
        1: 'Adam_sweep_1672475897_1',
        2: 'Adam_sweep_1672488572_2',
        3: 'Adam_sweep_1672489120_3',
        4: 'Adam_sweep_1672501329_4'}
# %%
