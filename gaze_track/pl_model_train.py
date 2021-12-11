from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *

from gaze_track.augmentations import DataAugmentationImage
from gaze_track.gaze_track_models import Gaze_Predictor

from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class Gaze_Track_pl(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
        super().__init__()
        self.save_hyperparameters(hparams)
        self.save_hyperparameters()

        self.swa_model = None
        self.network = Gaze_Predictor(self.hparams)
        
        self.learning_params = self.hparams["training"]
        self.augment = DataAugmentationImage(max_epochs = self.hparams["training"]["epochs"])

        self.heatmapLoss = HeatmapLoss()
        self.landmarks_loss = nn.MSELoss()
        self.gaze_loss = nn.MSELoss()

    def forward(self, x):
        gaze, heatmaps, landmarks_out = self.network(x)
        return gaze, heatmaps, landmarks_out


    def loss_calc(self, heatmaps_pred, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze):

        heatmap_loss = self.heatmapLoss(heatmaps_pred, heatmaps)
        landmarks_loss = self.landmarks_loss(landmarks_pred, landmarks)
        gaze_loss = self.gaze_loss(gaze_pred, gaze) * 100

        return torch.sum(heatmap_loss), landmarks_loss, gaze_loss


    def shared_step(self, batch):
        imgs = batch['img'].float()
        heatmaps = batch['heatmaps'].float()
        landmarks = batch['landmarks'].float()
        gaze = batch['gaze'].float()
        gaze_pred, heatmaps_pred, landmarks_pred = self(imgs)

        heatmaps_loss, landmarks_loss, gaze_loss = self.loss_calc(
                    heatmaps_pred, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze,
                    )
        return heatmaps_loss, landmarks_loss, gaze_loss


    def training_step(self, batch, batch_idx):
        batch['img'] = self.augment(batch['img'].float(), self.current_epoch)
        heatmaps_loss, landmarks_loss, gaze_loss = self.shared_step(batch)
        loss = (landmarks_loss + gaze_loss + heatmaps_loss) #* 10)
        
        self.log('train_loss_heatmap', heatmaps_loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_loss_landmarks', landmarks_loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_loss_gaze', gaze_loss, on_step=False, on_epoch=True, logger=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # prog_bar=True
      
        return loss


    def validation_step(self, batch, batch_idx):
        heatmaps_loss, landmarks_loss, gaze_loss = self.shared_step(batch)
        loss = (landmarks_loss + gaze_loss + heatmaps_loss) #* 10)
        
        self.log('vall_loss_heatmap', heatmaps_loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss_landmarks', landmarks_loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss_gaze', gaze_loss, on_step=False, on_epoch=True, logger=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True) # prog_bar=True
        
        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        heatmaps_loss, landmarks_loss, gaze_loss = self.shared_step(batch)
        loss = (landmarks_loss + gaze_loss + heatmaps_loss) #* 10)
        
        self.log('test_loss_heatmap', heatmaps_loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_loss_landmarks', landmarks_loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_loss_gaze', gaze_loss, on_step=False, on_epoch=True, logger=True)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True) # prog_bar=True
        
        return {'test_loss': loss}


    #copied
    def get_lr_inside(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def training_epoch_end(self, outputs):
        self.log('epoch_now', self.current_epoch, on_step=False, on_epoch=True, logger=True)
        (oppp) =  self.optimizers(use_pl_optimizer=True)
        self.log('lr_now', self.get_lr_inside(oppp), on_step=False, on_epoch=True, logger=True)



    def configure_optimizers(self):
        if self.learning_params["optimizer"] == "belief":
            optimizer =  AdaBelief(self.parameters(), lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                    weight_decouple = self.learning_params["weight_decouple"], 
                                    weight_decay = self.learning_params["weight_decay"], rectify = self.learning_params["rectify"])
        elif self.learning_params["optimizer"] == "ranger_belief":
            optimizer = RangerAdaBelief(self.parameters(), lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                       weight_decouple = self.learning_params["weight_decouple"],  weight_decay = self.learning_params["weight_decay"],)
        
        elif self.learning_params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_params["lr"])
        elif self.learning_params["optimizer"] == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_params["lr"])        

        if self.learning_params["add_sch"]:
            lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
	                                                                        max_lr=self.learning_params["lr"],
	                                                                        steps_per_epoch=0, #self.hparams.steps_per_epoch, #int(len(train_loader))
	                                                                        epochs=self.learning_params["epochs"],
	                                                                        anneal_strategy='linear'),
                        'name': 'lr_scheduler_lr',
                        'interval': 'step', # or 'epoch'
                        'frequency': 1,
                        }
            print("sch added")
            return [optimizer], [lr_scheduler]

        return optimizer