import torch

from train.utils import create_directory

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class ModelSaver():

    def __init__(self,path_to_save_pretrained_models, save_every_n_epoch=10, loss_min=True, debug=False) -> None:
        self.debug = debug
        self.path_to_save_pretrained_models = path_to_save_pretrained_models
        self.save_every_n_epoch = save_every_n_epoch
        self.loss_min = loss_min
        if not self.debug:
            create_directory(path_to_save_pretrained_models)

            if self.loss_min == True:
                self.best_loss = float('inf')
            else:
                self.best_loss = 0



    def update(self, loss, epoch, model):

        def check_better(loss):
            if self.loss_min:
                return self.best_loss > loss
            else:
                return self.best_loss < loss

        if not self.debug:
            if epoch % self.save_every_n_epoch == 0: #epoch == 0,10,20,... or best_loss_val > running_loss_val_total:
                torch.save(model.state_dict(), ('%s/model_%d.pth')%(self.path_to_save_pretrained_models, epoch))

            if check_better(loss):
                torch.save(model.state_dict(), ('%s/best_model.pth')%(self.path_to_save_pretrained_models))


    def save_final_model(self, model):
        if not self.debug:
            torch.save(model.state_dict(), ('%s/model_final.pth')%(self.path_to_save_pretrained_models))