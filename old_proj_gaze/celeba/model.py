from dependency import *


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Re_model(nn.Module):
    def __init__(self, re_dict):
      super(Re_model, self).__init__()
      self.feature_extractor = torchvision.models.mobilenet_v3_small() #out 576
      self.activation = self.feature_extractor.classifier[1] #Mish()
      self.feature_extractor.classifier = nn.Identity()
      
      self.norm = nn.Conv1d(re_dict["feature_out"], re_dict["feature_out"], 1)
      
      self.w1 = nn.Linear(in_features = re_dict["feature_out"], out_features = int(re_dict["feature_out"] * 1.5))
      self.dropout = nn.Dropout(re_dict["classificator_dropout"])
      self.w2 = nn.Linear(in_features = int(re_dict["feature_out"] * 1.5), out_features = re_dict["output_size"])
      
    def forward(self, x):
      x = self.feature_extractor(x)
      x = self.norm(x[:, :, None])[:, :, 0]
      x = self.activation(self.dropout(self.w1(x)))
      x = self.w2(x)

      return x

class Re_pl(pl.LightningModule):
    def __init__(self, re_dict, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
        super().__init__()
        self.save_hyperparameters(re_dict)
        self.save_hyperparameters()
        #self.hparams = hparams
        self.swa_model = None

        self.network = Re_model(self.hparams["model"])
        self.learning_params = self.hparams["training"]

        self.swa_mode = False

        self.criterion = nn.MSELoss()

    def forward(self, x):
        if not self.swa_mode:
            return self.network(x.float())#.float())
        else:
            return self.swa_model(x.float())#.float())
    

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
	                                                                        steps_per_epoch=self.hparams.steps_per_epoch, #int(len(train_loader))
	                                                                        epochs=self.learning_params["epochs"],
	                                                                        anneal_strategy='linear'),
                        'name': 'lr_scheduler_lr',
                        'interval': 'step', # or 'epoch'
                        'frequency': 1,
                        }
            print("sch added")
            return [optimizer], [lr_scheduler]

        return optimizer
    

    def training_step(self, batch, batch_idx):
        #also Manual optimization exist
        images, landmarks = batch['image'], batch['landmarks']
        landmarks = landmarks.view(landmarks.size(0),-1)
        
        predictions = self(images)

        loss = self.criterion(predictions.float(), landmarks.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # prog_bar=True
        return loss

    #copied
    def get_lr_inside(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def training_epoch_end(self, outputs):
        self.log('epoch_now', self.current_epoch, on_step=False, on_epoch=True, logger=True)
        (oppp) =  self.optimizers(use_pl_optimizer=True)
        self.log('lr_now', self.get_lr_inside(oppp), on_step=False, on_epoch=True, logger=True)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3095
        if self.learning_params["swa"] and (self.current_epoch >= self.learning_params["swa_start_epoch"]):
            if self.swa_model is None:
                (optimizer) =  self.optimizers(use_pl_optimizer=True)
                print("creating_swa")
                self.swa_model = AveragedModel(self.network)
                self.new_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr = self.learning_params["swa_lr"])
            # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            self.swa_model.update_parameters(self.network)
            self.new_scheduler.step()
    

    def change_for_swa(self, loader):
        print("will it work?")
        torch.optim.swa_utils.update_bn(loader, self.swa_model)
        self.swa_mode = True
        return


    def validation_step(self, batch, batch_idx):

        images, landmarks = batch['image'], batch['landmarks']
        landmarks = landmarks.view(landmarks.size(0),-1)
        
        predictions = self(images)
        loss = self.criterion(predictions.float(), landmarks.float())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True) # prog_bar=True

        return {'val_loss': loss}