from os import sys, path
from tkinter import X
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *
from basem.subblocks import Swish, Mish

from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from gaze_track.dataset_mpi_utils import angularError
from gaze_track.augmentations import DataAugmentationImage

from basem.modules import Modified_Encoder_Layer, Encoder_Block
from einops import rearrange

def add_token(network, d_model_emb, n_of_learn_params = 1):
  #nn.Parameter(torch.randn(1, n_of_learn_params, feature_extract_hparams["d_model_emb"]))
  old_tokens = network.feature_extcractor.zero_class_token 
  added_token = nn.Parameter(torch.randn(1, n_of_learn_params, d_model_emb))
  network.feature_extcractor.zero_class_token = torch.cat((added_token, old_tokens), dim=1)
  network.number_of_learn_params += n_of_learn_params
  return network



class Updated_Feature_Encoder(nn.Module):
  def __init__(self, feature_model, feature_extract_hparams_, new_token = False):
    super().__init__()
    #copy everything and block, also can add maybe token here

    self.patch_embedding = feature_model.patch_embedding
    self.dropout = feature_model.dropout
    self.zero_class_token = feature_model.zero_class_token
    self.positional_embedding = feature_model.positional_embedding
    self.encoder = feature_model.encoder
    self.feature_extcractor = feature_model.feature_extcractor
    self.number_of_learn_params = feature_model.number_of_learn_params
    self.image_extcractor = feature_model.image_extcractor
    self.add_train_land = feature_model.add_train_land
    self.get_landmarks = feature_model.get_landmarks
    self.patch_size = feature_model.patch_size

    self.new_token = new_token

    if new_token:
      self.added_token = nn.Parameter(torch.randn(1, 1, feature_extract_hparams_["d_model_emb"]))
      self.number_of_learn_params += 1


    feature_extract_hparams = dict.copy(feature_extract_hparams_)
    
    if feature_extract_hparams["encoder_type"] == "evolved":
      self.added_encoder = Modified_Encoder_Layer(feature_extract_hparams["encoder_params"])
    elif feature_extract_hparams["encoder_type"] == "transformer":
      feature_extract_hparams["encoder_params"]["layers_number"] = 1
      self.added_encoder = Encoder_Block(feature_extract_hparams["encoder_params"])

  def forward(self, x):
  # copy every subpart of the model and go, just let landmark go, but change feature extraction
    x = rearrange(x, 'b c (h p) (w pd) -> b (h w) (p pd c)', p = self.patch_size, pd = self.patch_size)

    x = self.patch_embedding(x)
    x = self.dropout(x)

    b, n, d = x.shape

    if self.new_token:
      added_token = torch.cat((self.added_token, self.zero_class_token), dim=1)
    else:
      added_token = self.zero_class_token

    zero_class_token = torch.repeat_interleave(added_token, repeats = b, dim=0)

    x = torch.cat((zero_class_token, x), dim=1)

    x = self.positional_embedding(x)
    x = self.encoder(x)


    land = self.image_extcractor(x[:, self.number_of_learn_params:])

    land = self.add_train_land(land)

    # c = self.channels, p = self.patch_size, h = self.im_size_h, w = self.im_size
    land = self.get_landmarks(land)   

    x = self.added_encoder(x)
    feature_vector = self.feature_extcractor(x[:, 0:self.number_of_learn_params])


    return land, feature_vector


class Updated_Gaze(nn.Module):
    def __init__(self, old_gaze, d_model_emb, gaze_size, mlp_drop):
        super().__init__()
        self.old_gaze = old_gaze
        self.new_gaze = nn.Sequential(nn.Linear(d_model_emb, d_model_emb),
                                      Mish(), # Swich
                                      nn.Dropout(mlp_drop),
                                      nn.Linear(d_model_emb, gaze_size),
                                    )
        self.combine_gaze = nn.Linear(gaze_size * 2, gaze_size)
    def forward(self, x):
        x_1 = self.old_gaze(x)
        x_2 = self.new_gaze(x)
        x = torch.cat((x_1, x_2), 1)
        x = self.combine_gaze(x)
        return x


class MPI_Gaze_Track_pl(pl.LightningModule):
  def __init__(self, hparams, model = None, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
    super().__init__()
    self.save_hyperparameters(hparams)
    self.save_hyperparameters()

    self.swa_model = None

    self.network = model.network

    # work only for one trainable token, rewrite when will use more
    # I do not lock landmarks tokens, so could be a problem
    # but a possible solution could be param[:, 1:].detach()
    # but not sure so unlocking all
    # I think this weights will not change anyway

    unlock_tokens = self.hparams["unlock_tokens"]

    for name, param in self.network.named_parameters():
        if param.requires_grad and 'landmarks_extract' in name:
            param.requires_grad = False

        if self.hparams["lock_main_weights"]:
            if param.requires_grad and 'feature_extcractor' in name:
                #
                if unlock_tokens and ('zero_class_token' in name):
                    print("unlock tokens")
                    continue
                #
                param.requires_grad = False


    d_model_emb = model.hparams["feature_extractor_hparams"]["d_model_emb"]
    gaze_size = model.hparams["gaze_size"]
    number_of_learn_params = model.hparams["feature_extractor_hparams"]["number_of_learn_params"]
    if self.hparams["add_encoder_for_gaze"]:
      self.network.feature_extcractor = Updated_Feature_Encoder(self.network.feature_extcractor, 
                                        model.hparams["feature_extractor_hparams"], self.hparams["add_token"])
      if self.hparams["add_token"]:
        number_of_learn_params += 1

    if self.hparams["updated_gaze"]:
        self.network.gaze_mlp = Updated_Gaze(self.network.gaze_mlp, number_of_learn_params * d_model_emb, 
                                            gaze_size, self.hparams["mlp_drop"])

    if self.hparams["new_gaze_weights"]:
      d_model_emb = d_model_emb * number_of_learn_params
      self.network.gaze_mlp = nn.Sequential(nn.Linear(d_model_emb, d_model_emb),
                                      Mish(), # Swich
                                      nn.Dropout(self.hparams["mlp_drop"]),
                                      nn.Linear(d_model_emb, gaze_size),
                                    )
    self.augment = DataAugmentationImage(max_epochs = self.hparams["epochs"])

    if self.hparams["loss_function"] == "mse":
      self.gaze_loss = nn.MSELoss()
    elif self.hparams["loss_function"] == "mae":
      self.gaze_loss = nn.L1Loss()
    self.gaze_loss_val = nn.MSELoss()

    self.learning_params = self.hparams["training"]


  def tranform_into_actual_coor(self, landmarks_pred):
    landmarks_pred = (landmarks_pred + 0.5)
    landmarks_pred[:,:, 0] *= self.hparams["size"][0]
    landmarks_pred[:,:, 1] *= self.hparams["size"][1]
    return landmarks_pred


  def forward(self, x):
    gaze, heatmaps, landmarks_out = self.network(x)
    return gaze, heatmaps, landmarks_out


  def training_step(self, batch, batch_idx):
    
    imgs, gaze = batch
    imgs, gaze = imgs.float(), gaze.float()

    imgs = self.augment(imgs, self.current_epoch)

    gaze_pred, _, _ = self(imgs)

    loss = self.gaze_loss(gaze_pred, gaze) * 100
    self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # prog_bar=True

    ang_error = angularError(gaze, gaze_pred)
    self.log('train_angular_error', ang_error, on_step=False, on_epoch=True, logger=True)

    return loss


  def validation_step(self, batch, batch_idx):

    imgs, gaze = batch
    imgs, gaze = imgs.float(), gaze.float()

    imgs = self.augment(imgs, self.current_epoch)

    gaze_pred, _, _ = self(imgs)

    loss = self.gaze_loss_val(gaze_pred, gaze) * 100
    self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True) # prog_bar=True
    
    ang_error = angularError(gaze, gaze_pred)
    self.log('val_angular_error', ang_error, on_step=False, on_epoch=True, logger=True)

    return {'val_loss': loss}


  def test_step(self, batch, batch_idx):

    imgs, gaze = batch
    imgs, gaze = imgs.float(), gaze.float()

    imgs = self.augment(imgs, self.current_epoch)

    gaze_pred, _, _ = self(imgs)

    loss = self.gaze_loss_val(gaze_pred, gaze) * 100
    
    self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True) # prog_bar=True

    ang_error = angularError(gaze, gaze_pred)
    self.log('test_angular_error', ang_error, on_step=False, on_epoch=True, logger=True)

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
        optimizer =  AdaBelief(filter(lambda p: p.requires_grad, self.parameters()),
                               lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                weight_decouple = self.learning_params["weight_decouple"], 
                                weight_decay = self.learning_params["weight_decay"], rectify = self.learning_params["rectify"])
    elif self.learning_params["optimizer"] == "ranger_belief":
        optimizer = RangerAdaBelief(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                    weight_decouple = self.learning_params["weight_decouple"],  weight_decay = self.learning_params["weight_decay"],)
    
    elif self.learning_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                                     lr=self.learning_params["lr"])
    elif self.learning_params["optimizer"] == "adamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                      lr=self.learning_params["lr"])        

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