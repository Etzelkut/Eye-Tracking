from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *

import torchvision.transforms.functional as TF
from torchvision import transforms

import glob
from dataset_mpi_utils import read_files_mpi_val, process_mpi_files, devide_val


class MPI_Preprocess(nn.Module):
  @torch.no_grad()  # disable gradients for effiency
  def forward(self, img, new_size):

    x, y = new_size
    img = TF.to_tensor(img).float() # CxHxW

    x_ = int((256/224) * x)
    y_ = int((256/224) * y)
    output_size = (x_, y_)
    img = TF.resize(img, output_size)

    top =  int((x_ - x)/2)
    left =  int((y_ - y)/2)

    img = img[:, top: top + x, left: left + y]

    return img


class MPIIGaze(Dataset):

  def __init__(self, images, gazes, val = False, transform = MPI_Preprocess(), size = (96, 160)):
    super().__init__()
    self.images = images
    self.gazes = gazes
    self.val = val

    self.size = size
    self.transform = transform

    self.norm = transforms.Normalize([0.5], [0.5])

  def __len__(self):
    return len(self.gazes)

  def __getitem__(self, idx):
    image = self.transform(self.images[idx], self.size)
    
    if self.val:
      image = self.norm(image)
    
    return image, self.gazes[idx]



class Dataset_mpi_pl(pl.LightningDataModule):
  def __init__(self, datahparams):
    super().__init__()
    self.save_hyperparameters(datahparams)

  def prepare_data(self):
    print("can add download here")
  
  def setup(self):
    path = self.hparams["main_path"] + '/Data/Normalized'

    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    
    eval_files = glob.glob(self.hparams["main_path"] + '/Evaluation Subset/sample list for eye image/*.txt')
    eval_full_names = read_files_mpi_val(eval_files, path)

    images, gazes, images_index_name = process_mpi_files(listOfFiles)
    images_train, gazes_train, images_val, gazes_val = devide_val(images, gazes, images_index_name, eval_full_names)

    dataset_train = MPIIGaze(images_train, gazes_train)
    N = len(dataset_train)
    vn = int(N*0.1)
    tn = N - vn
    self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset_train, (tn, vn))

    self.dataset_test = MPIIGaze(images_val, gazes_val, val = True)

    print(len(self.dataset_train), len(self.dataset_val), len(self.dataset_test))


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
    test = DataLoader(self.dataset_test, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], 
                            shuffle=False, 
                            )
    return test