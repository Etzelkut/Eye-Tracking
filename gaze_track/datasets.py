from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import glob

from basem.basic_dependency import *
from gaze_track.dataset_util import preprocess_unityeyes_image, get_heatmaps
from gaze_track.augmentations import Rescale, CenterCrop, RandomCrop, transforms, Preprocess

import cv2
import json
import torchvision.transforms.functional as TF

class UnityEyesDataset(Dataset):
  def __init__(self, img_dir, output_size, transform=None, grayscale = True, val = True, halfing = False):
    self.transform = transform
    if img_dir is None:
        img_dir = './gazeset/imgs'

    self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    self.json_paths = []
    for img_path in self.img_paths:
        idx = os.path.splitext(os.path.basename(img_path))[0]
        self.json_paths.append(os.path.join(img_dir, f'{idx}.json'))
    
    self.grayscale = grayscale
    self.output_size = output_size
    self.val = val
    self.halfing = halfing
    #idxs_aranged = np.arange(len(self.img_paths))
    np.random.seed(42)

    idxs_train = sorted(np.random.choice(len(self.img_paths), size=int(len(self.img_paths) * 0.8), replace=False))
    mask_val_test = np.ones(len(self.img_paths), dtype=bool)
    mask_val_test[idxs_train] = False
    #idxs_aranged_valtest = idxs_aranged[mask_val_test]

    if val:
      self.img_paths = np.array(self.img_paths)[mask_val_test]
      self.json_paths = np.array(self.json_paths)[mask_val_test]
      self.resize = Rescale()
      self.crop = CenterCrop(output_size)
      print("val")
      self.norm = transforms.Normalize([0.5], [0.5])
    else:
      self.img_paths = np.array(self.img_paths)[idxs_train]
      self.json_paths = np.array(self.json_paths)[idxs_train]
      self.resize = Rescale()
      self.crop = RandomCrop(output_size)
      print("train")

  
  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    full_img = cv2.imread(self.img_paths[idx])
    #print("im read")
    #print(np.min(full_img))
    #print(np.max(full_img))
    
    with open(self.json_paths[idx]) as f:
        json_data = json.load(f)

    sample = {}
    
    eye_sample = preprocess_unityeyes_image(full_img, json_data)

    #print("im preprocess_unityeyes_image")
    #print(np.min(eye_sample['img']))
    #print(np.max(eye_sample['img']))


    if self.transform:
      eye_sample['img'] = self.transform(eye_sample['img'])

      #print("after preproccess")
      #print(torch.max(eye_sample['img']))
      #print(torch.min(eye_sample['img']))

    if self.grayscale:
      eye_sample['img'] = TF.rgb_to_grayscale(eye_sample['img'])
      #print("after grayacale")
      #print(torch.max(eye_sample['img']))
      #print(torch.min(eye_sample['img']))
    
    if self.val:
      #print("val")
      x, y = self.output_size
      x = int((256/224) * x)
      y = int((256/224) * y)
      output_size = (x, y)
      eye_sample = self.resize(eye_sample, output_size)

      eye_sample = self.crop(eye_sample)

      eye_sample['img'] = self.norm(eye_sample['img'])
    
    else:
      #print("train")
      x1, y1 = eye_sample['img'].shape[-2:]
      x2, y2 = self.output_size
      x2 = random.randint(int(x2/2), x1)
      y2 = int(x2/x1 * y1)
      random_size = (x2, y2)
      #print("random size: ", random_size)
      eye_sample = self.resize(eye_sample, random_size)

      x, y = self.output_size
      x = int((256/224) * x)
      y = int((256/224) * y)
      output_size = (x, y)
      eye_sample = self.resize(eye_sample, output_size)

      eye_sample = self.crop(eye_sample)

    halfed = 1
    if self.halfing:
      eye_sample['landmarks'] = eye_sample['landmarks'] * np.array([0.5, 0.5])
      eye_sample['landmarks'] = eye_sample['landmarks'].astype(np.float32)
      halfed = 0.5

    heatmaps = get_heatmaps(int(eye_sample['img'].shape[-1] * halfed),  int(eye_sample['img'].shape[-2] * halfed), eye_sample['landmarks'])
    eye_sample["heatmaps"] = heatmaps
    sample.update(eye_sample)
    return sample



class Dataset_Unity_pl(pl.LightningDataModule):
  def __init__(self, datahparams):
    super().__init__()
    self.save_hyperparameters(datahparams)

  def prepare_data(self):
    print("can add download here")
    
  def setup(self):
    
    if "halfing" not in self.hparams:
      halfing = False
    else:
      halfing = self.hparams["halfing"]
      print("halfing is ", self.hparams["halfing"])

    self.dataset_train = UnityEyesDataset(self.hparams["img_dir"], self.hparams["im_size"], transform=Preprocess(), 
                                          grayscale = self.hparams["grayscale"], val = False, halfing=halfing)
    dataset_val = UnityEyesDataset(self.hparams["img_dir"], self.hparams["im_size"], transform=Preprocess(), 
                                    grayscale = self.hparams["grayscale"], val = True, halfing=halfing)

    N = len(dataset_val)
    vn = int(N/2)
    tn = N - vn
    self.dataset_test, self.dataset_val = torch.utils.data.random_split(dataset_val, (tn, vn))


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
