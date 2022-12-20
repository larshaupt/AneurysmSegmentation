#%%
from train.utils import *
from train.metrics import MetricesStruct
from train.callbacks import EarlyStopping

import wandb
import time
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from types import ModuleType
import matplotlib
import pandas as pd
import os
matplotlib.use("Agg")

from torch.utils.tensorboard import SummaryWriter
tb_dir = "/srv/beegfs02/scratch/brain_artery/data/training/logging/tensorbord"
wandb_dir = "/srv/beegfs02/scratch/brain_artery/data/training/logging/wandb"
#%%
def wandb_visualize(data, target, pred, name, table, num=1):
    # WandB Visualization
    for index in range(min(data.shape[0], pred.shape[0], target.shape[0], num)):
        #assert data.dim == 5 and target.dim == 5 and pred.dim == 5
        data_el, target_el, pred_el = data[index,...], target[index,...], pred[index,...]
        slices_ind = find_slices(target_el, num, ratio_pos=0.8)
        
        pred_el = binarize(pred_el) # binarize prediction
        target_el = binarize(target_el)  # binarize prediction
        convert_to_img = lambda x: x.to('cpu').detach().numpy()
        data_el, target_el, pred_el = convert_to_img(data_el), convert_to_img(target_el), convert_to_img(pred_el)
        
        for slice_no in sorted([el[-1] for el in slices_ind]):
            data_slice = data_el[...,slice_no]
            target_slice = target_el[...,slice_no]
            pred_slice = pred_el[...,slice_no]
            data_slice = (data_slice)/np.max(data_slice) # wandb input needs to be a float between 0 and 2
            table.add_data(name[0], str(slice_no), wandb.Image(data_slice), wandb.Image(target_slice), wandb.Image(pred_slice), wb_mask(data_slice, pred_slice[0,:,:]))
    
        

def wandb_init(exp_config, model, exp_name = "", wandb_log_grad=False):

    """     wandb.config = {
        "learning_rate": exp_config.learning_rate,
        "epochs": exp_config.number_of_epoch,
        "batch_size": exp_config.batch_size,
        "batch_size_val": exp_config.batch_size_val,
        "num_classes" : exp_config.num_classes,
        "lambda_loss" : exp_config.lambda_loss,
        "model" : exp_config.model,
        "dataset" : exp_config.dataset,
        "fold_id" : exp_config.fold_id,
        "timestamp" : exp_config.timestamp,
        "experiment_name" : exp_config.experiment_name,
        "criterion_loss" : exp_config.criterion_loss,
        "criterion_metric" : exp_config.criterion_metric,
        "tf_train" : exp_config.tf_train,
        "tf_test" : exp_config.tf_test,
        "tf_val" : exp_config.tf_val,
        "shuffle_train" : exp_config.shuffle_train, 
        "shuffle_validation" : exp_config.shuffle_validation, 
        "shuffle_test" : exp_config.shuffle_test, 
        "training_dataset" : exp_config.training_dataset, 
        "validation_dataset" : exp_config.validation_dataset, 
        "test_dataset" : exp_config.test_dataset, 
        "num_training_files" : exp_config.num_training_files, 
        "positive_weight" : exp_config.positive_weight,
        "patch_size" : exp_config.patch_size,
        "augment_probability" : exp_config.augment_probability,
    } """
    if exp_name == "":
        exp_name = exp_config.experiment_name

    # Sorry for the one-liner. Sorts out all attributes from exp_config that cannot be read into a json
    log_dict = {key:exp_config.__dict__[key] for key in exp_config.__dict__.keys() if key[:2] != '__' and not isinstance(exp_config.__dict__[key], ModuleType) and key not in ['Tensor', 'Callable']}
    

    if hasattr(exp_config, "tags"):
        tags = exp_config.tags
    else:
        tags = ["default"]

    if hasattr(exp_config, "project_name"):
        project_name = exp_config.project_name
    else:
        project_name = "USZ_prediction"
    

    wandb.init(project=project_name, entity="lhauptmann", name=exp_name, config=log_dict, tags=tags, dir=wandb_dir)
    #wandb.run.log_code(".") # to keep track of changes in the code across multiple runs (check how to use)
    if wandb_log_grad:
        wandb.watch(model, log="all")
    columns = ["filename", "slice_no", "image", "ground_truth", "prediction", "masked_prediction"]
    wandb_table = wandb.Table(columns=columns)

    return wandb_table

def wb_mask(bg_img, mask):
    return wandb.Image(bg_img, masks={
    "ground truth" : {"mask_data" : mask, "class_labels" : {0: "background", 1: "mask"} }})


def tensorboard_log_stats(writer, stats_dict, epoch):
    for key in stats_dict:
        writer.add_scalar(key, stats_dict[key], epoch)

    
def save_images(data, target, pred, name, epoch, exp_config, data_set = "val", save_overlap=True, num=1):
    for index in range(min(data.shape[0], pred.shape[0], target.shape[0], num)): # for each file in batch
        save_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/Training/predictions/%s/' %(exp_config.experiment_name)
        save_path_train = os.path.join(save_path, 'train_monitoring')
        save_path_epoch = os.path.join(save_path_train, str(epoch))
        save_path_set = os.path.join(save_path_epoch, data_set)
        save_path_example = os.path.join(save_path_set, name[index])

        for _path in [save_path, save_path_train,save_path_epoch,save_path_set, save_path_example]:
            if not os.path.exists(_path):
                os.mkdir(_path)

        data_el, target_el, pred_el = data[index,...], target[index,...], pred[index,...]

        slices_ind = find_slices(target_el, num, ratio_pos=0.8)
        pred_el = binarize(pred_el) # binarize prediction
        target_el = binarize(target_el)  # binarize prediction
        convert_to_img = lambda x: (x.to('cpu').detach().numpy()*255).astype(np.uint8)
        data_el, target_el, pred_el = convert_to_img(data_el), convert_to_img(target_el), convert_to_img(pred_el)
        #print(list(np.linspace(int(data_el.shape[-1]/10), int(data_el.shape[-1]*9/10), 6, dtype=int)) + [z_slice_seleceted])
        for slice_no in sorted([el[-1] for el in slices_ind]):
            save_path_slice_data = os.path.join(save_path_example, str(slice_no)+'_data.png')
            save_path_slice_target = os.path.join(save_path_example, str(slice_no)+'_target.png')
            save_path_slice_pred = os.path.join(save_path_example, str(slice_no)+'_pred.png')

            data_slice = data_el[:,:,:,slice_no]
            target_slice = target_el[:,:,:,slice_no]
            pred_slice = pred_el[:,:,:,slice_no]
            
            for (file, path) in [(data_slice, save_path_slice_data), (target_slice, save_path_slice_target), (pred_slice,save_path_slice_pred)]:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(file)
                
                fig.savefig(path)
                plt.close(fig)

            if save_overlap:
                save_path_overlap = os.path.join(save_path_example, str(slice_no)+'_overlap.png')
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(data_slice)
                ax.imshow(pred_slice, cmap="jet", alpha=0.5)
                metric_score = exp_config.criterion_metric['Dice'](pred_slice, target_slice)
                ax.text(0,0, "Dice=" + str(metric_score), backgroundcolor="white")
                fig.savefig(save_path_overlap)
                plt.close(fig)



def tb_visualize(writer, data, target, pred, name):
    # Tensorboard Visualization
    for index in range(min(data.shape[0], pred.shape[0], target.shape[0])):
        data_el, target_el, pred_el, name_el  = data[index], target[index], pred[index], name[index]
        slices_ind = find_slices(target_el, 5, ratio_pos=0.8)
        
        pred_el = binarize(pred_el) # binarize prediction
        target_el = binarize(target_el)  # binarize prediction
        convert_to_img = lambda x: x.to('cpu')
        data_el, target_el, pred_el = convert_to_img(data_el), convert_to_img(target_el), convert_to_img(pred_el)

        for slice_no in sorted([el[-1] for el in slices_ind]):
            reverse_onehot = lambda x : torch.unsqueeze(x.argmax(0), 0)
            data_slice = data_el[...,slice_no]
            target_slice = reverse_onehot(target_el[...,slice_no])
            pred_slice = reverse_onehot(pred_el[...,slice_no])

            img_grid = make_grid([data_slice, target_slice, pred_slice])
            writer.add_image(name_el, img_grid)

                
def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
class TrainLogger():
    def __init__(self, 
            exp_config, 
            model, 
            exp_name, 
            path_to_save_pretrained_models, 
            wandb_log=True, 
            tb_log = True, 
            save_images=False, 
            tb_visualize=False, 
            wandb_visualize=False, 
            wandb_log_grad = False,
            debug=False, 
            max_images = 2,
            ) -> None:

        self.run = not debug
        self.exp_config = exp_config
        self.model = model
        self.exp_name = exp_name
        self.path_to_save_pretrained_models = path_to_save_pretrained_models
        self.wandb_log = self.run and wandb_log
        self.tb_log = self.run and tb_log
        self.save_images = self.run and save_images
        self.wandb_visualize = self.run and wandb_visualize
        self.tb_visualize = self.run and tb_visualize
        self.wandb_table = None
        self.tb_writer = None
        self.epoch_counter = 0
        self.current_epoch = 0
        self.max_images = max_images
        self.wandb_log_grad = wandb_log_grad

        self.score_logger = ScoreLogger()

        if debug:
            os.environ['WANDB_SILENT']="true"
        else:
            os.environ['WANDB_SILENT']="false"
        

        if self.wandb_log:
            self.wandb_table = wandb_init(exp_config, model, exp_name, wandb_log_grad=self.wandb_log_grad)

        if self.tb_log:
            self.tb_writer = SummaryWriter(comment = '_' + exp_name, log_dir = tb_dir)

    

    def visualize_val(self,data, target, pred, name,epoch , num=1):
        if self.wandb_visualize:
            if self.epoch_counter < self.max_images:
                wandb_visualize(data, target, pred, name, self.wandb_table, num=num)
        if self.save_images:
            save_images(data, target, pred, name, epoch, self.exp_config, data_set="val", num=num)
        if self.tb_visualize:
            tb_visualize(self.tb_writer, data, target, pred, name)
        self.epoch_counter +=1

    def log(self, log_dict, epoch, verbose=True):
        self.current_epoch = epoch
        self.score_logger.log(log_dict, epoch)
        if self.wandb_log:
            try:
                wandb.log(log_dict, step=self.current_epoch)
            except:
                print('Error in Wandb Logging. Skipping...')
        if self.tb_log:
            tensorboard_log_stats(self.tb_writer,log_dict, epoch)

        if verbose:
            self.score_logger.print_epoch_scores()
    
    def end_of_epoch(self):
        if self.wandb_visualize:   
            try:   
                wandb.log({"Prediction": self.wandb_table}, step=self.current_epoch)
                best_score, best_i =  self.score_logger.get_best_score("val_Dice", max=True, return_epoch=True)
                wandb.run.summary["best_score"] = best_score
                wandb.run.summary["best_epoch"] = best_i
            except:
                print('Error in Wandb Logging. Skipping...')

            self.epoch_counter = 0

        if self.tb_log:
            self.tb_writer.flush()

    def end_of_training(self):
        if self.tb_log:
            self.tb_writer.close()
        if self.wandb_log:
            try:
                best_score, best_i =  self.score_logger.get_best_score("val_Dice", max=True, return_epoch=True)
                wandb.run.summary["best_score"] = best_score
                wandb.run.summary["best_epoch"] = best_i
                wandb.run.summary["last_epoch"] = self.current_epoch
                wandb.finish(quiet=True)
            except:
                print('Error in Wandb Logging. Skipping...')
        self.score_logger.save_log(self.path_to_save_pretrained_models)

    def get_score_logger(self):
        return self.score_logger
        

    
class ScoreLogger():
    def __init__(self):
        self.log_dicts = pd.DataFrame(data={})
        self.current_epoch = 0

    def log(self, log_dict:dict, epoch:int):
        if epoch in self.log_dicts:
            print(f"Warning: Epoch {epoch} is aready logged.")

        epoch_df = pd.DataFrame(data=log_dict, index=[epoch])
        self.log_dicts = pd.concat((self.log_dicts, epoch_df))
        self.current_epoch = epoch

    def get_best_score(self, metric:str="val_Dice", max=True, return_epoch=True):
        if max:
            best_i = self.log_dicts[metric].argmax()
        else:
            best_i = self.log_dicts[metric].argmin()

        best_score = self.log_dicts[metric].iloc[best_i]
        if return_epoch:
            return best_score, best_i
        else: 
            return best_score

    def get_log(self):
        return self.log_dicts


    def print_epoch_scores(self):
        output = 'epoch:%d - ' %(self.current_epoch)
        for col in self.log_dicts.columns:
            output += f"{col}: {self.log_dicts[col].iloc[self.current_epoch]:.5f}  "

        print(output)

    def save_log(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.log_dicts.to_csv(os.path.join(path, "training_log.csv"))

    def best_epoch(self, metric:str = "val_loss"):
        return self.log_dicts[metric].iloc[self.current_epoch] == self.log_dicts[metric].max()

class Timer():
    def __init__(self) -> None:
        self.times = []
        self.desciptions = []
        self.take_time('Start')


    def take_time(self, description:str=''):
        stop_time = time.time()
        self.times.append(stop_time)
        self.desciptions.append(description)
        if len(self.times) > 1:
            print(f'Time for {self.desciptions[-2]}: {self.times[-1] - self.times[-2]}')

    def analyze_times(self):
        assert len(self.times) == len(self.desciptions)
        for i in range(len(self.times)-1):
            print(f'Time for {self.desciptions[i]}: {self.times[i] - self.times[i+1]}')






#%%
def test_logger(logger):
    log_dict = {"val_Dice":0.8,
                "train_dice": 0.9}
    logger.update(log_dict)
    print(logger.get_best_score("val_Dice"))
# %%
