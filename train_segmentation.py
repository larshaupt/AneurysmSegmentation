# %%
from train.utils import *
from train_func import train_segmentation_network
from data import data_loader

from importlib.machinery import SourceFileLoader
import pdb
from train.options import Options




# %%

def main(exp_config, exp_cont = ""):

    """Main function that trains a segmentation network.

    Args:
    - exp_config (object): object containing all configuration parameters for training
    - exp_cont (str, optional): string representation of input arguments. Default value is an empty string.

    Returns:
    None
    """
    
    log = train(exp_config, fold_id = exp_config.fold_id,  exp_cont = exp_cont)
        
    scores = log.get_best_score("val_Dice", max=True, return_epoch=False)
    best_epochs = log.get_best_score("val_Dice", max=True, return_epoch=True)[1]
    print("Dice: ", scores, " Best Epochs: ", best_epochs)
    print("Mean Dice", scores)


def train(exp_config, fold_id, exp_cont = ""):
    
    """Trains a segmentation network.

    Args:
    - exp_config (object): object containing all configuration parameters for training
    - fold_id (int): fold identifier used to load a specific split of data
    - exp_cont (str, optional): string representation of input arguments. Default value is an empty string.

    Returns:
    - train_log (object): object containing training log information
    """

    experiment_name = f"{exp_config.experiment_name}_{fold_id}"

    print(f"\n############################################################################\nRunning {experiment_name}\n############################################################################\n")

    # =====================
    # Define network architecture
    # =====================    
    model = exp_config.model
    device = exp_config.device
    optimizer = exp_config.optimizer
    model.to(device)

    
    # =========================
    # Load source dataset
    # =========================

    split_dict = load_split(exp_config.path_split, fold_id)
    source_train_loader, source_val_loader, source_test_loader = data_loader.load_datasets(
                        exp_config = exp_config,
                        batch_size = exp_config.batch_size,
                        batch_size_test = exp_config.batch_size_val,
                        path_data = exp_config.path_data,
                        path_data_extra = exp_config.path_data_extra,
                        tf_train = exp_config.tf_train,
                        tf_val = exp_config.tf_val,
                        split_dict = split_dict, 
                        train_val_test = (exp_config.training_dataset, exp_config.validation_dataset, exp_config.test_dataset),
                        reduce_number = exp_config.num_training_files,
                        num_workers = exp_config.num_workers,
                        norm_percentile = exp_config.norm_percentile,
                        normalization = exp_config.normalization,
                        )

    
    # =========================
    # Train segmentation network
    # =========================
    train_log = train_segmentation_network(
                        exp_config, 
                        model,
                        optimizer, 
                        source_train_loader, 
                        source_val_loader, 
                        source_test_loader, 
                        path_to_save_pretrained_models = os.path.join(exp_config.path_to_save_pretrained_models , str(fold_id)),
                        device = exp_config.device,
                        exp_cont = exp_cont,
                        exp_name = experiment_name)
    
    return train_log
#%%
if __name__ == '__main__':

    options = Options(mode='train')
    exp_config = options.get_opt()


    if not exp_config.debug:
        options.save_params(exp_config.path_to_save_pretrained_models)

    main(exp_config=exp_config, exp_cont = options.print_input_arguments())



# %%
