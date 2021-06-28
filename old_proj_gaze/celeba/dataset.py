from dependency import *
from data_transform import *

class Reader(Dataset):

    #creates the generator taking the raw data and a batch size
    def __init__(self, generalFolder, transform = None):

      folder = generalFolder + 'img_align_celeba/img_align_celeba'
      allFiles = [folder + "/" + f for f in sorted(os.listdir(folder))]
      featureLocations = pd.read_csv(generalFolder + 'list_landmarks_align_celeba.csv')

      assert len(allFiles)==len(featureLocations)

      self.files = allFiles
      self.locations = np.array(featureLocations)[:,1:]   #5 coordinate pairs

      self.transform = transform
 
    def __len__(self):
      return len(self.files)
    
    #gets one batch respective to the passed index    
    def __getitem__(self, idx):
      im = io.imread(self.files[idx])
      loc = np.array(self.locations[idx][0:4], dtype=np.float64).reshape((-1,2))

      part = {}
      part["image"] = im
      part["landmarks"] = loc
      if self.transform:
          part = self.transform(part)
      #loc_torch = torch.from_numpy(loc).view(-1, 2)
      return part

class Dataset_CelebA(pl.LightningDataModule):
    def __init__(self, conf, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
      super().__init__()
      self.hparams = conf
      self.transform_train = transforms.Compose([
                               Rescale(self.hparams["scale_im"]),
                               Hflip(self.hparams["flip_chance"]),
                               Rotate(self.hparams["rotate_chance"]),
                               ToTensor(),
                               NormTensor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
      
      self.transform_test = transforms.Compose([Rescale(self.hparams["scale_im"]),
                               ToTensor(),
                               NormTensor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
      
    def prepare_data(self):
      print("can add download here")
    
    def setup(self):
      dataset = Reader(self.hparams["generalFolder"], self.transform_train)
      size_of_main = len(dataset)
      train_size = int(size_of_main*0.9)
      val_size = size_of_main - train_size
      self.dataset_train, self.dataset_val  = torch.utils.data.random_split(dataset, 
                                              [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42)
                                              )
      self.test = None

    def train_dataloader(self):
      data_train = DataLoader(self.dataset_train, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], 
                              shuffle=self.hparams["dataloader_shuffle"], 
                              )
      return data_train

    def val_dataloader(self):
      val = DataLoader(self.dataset_val, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], 
                              shuffle=False, 
                              )
      return val

    def test_dataloader(self):
      test = self.dataset_test
      return test