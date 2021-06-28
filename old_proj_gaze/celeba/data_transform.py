from dependency import *

def rotate(image, landmarks, angle):
  img_shape = image.shape
  landmarks = landmarks / np.array([img_shape[1], img_shape[0]])
  transformation_matrix = np.array([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

  landmarks = landmarks - 0.5
  new_landmarks = np.matmul(landmarks, transformation_matrix)
  new_landmarks = new_landmarks + 0.5
  
  new_landmarks = new_landmarks * np.array([img_shape[1], img_shape[0]])

  image = Image.fromarray(image)
  image = TF.rotate(image, angle)
  image = np.array(image)

  return image, new_landmarks

def hflip(image, landmarks):
  image = Image.fromarray(image)
  image = TF.hflip(image)
  image = np.array(image)
  landmarks[:, 0] =  image.shape[1]-landmarks[:, 0] - 1
  return image, landmarks


def resize(image, landmarks, output_size = (224, 224)):
  h, w = image.shape[:2]
  if isinstance(output_size, int):
      if h > w:
          new_h, new_w = output_size * h / w, output_size
      else:
          new_h, new_w = output_size, output_size * w / h
  else:
      new_h, new_w = output_size

  new_h, new_w = int(new_h), int(new_w)
  
  image = transform.resize(image, (new_h, new_w), preserve_range=True)
  image = image.astype(np.uint8)  #         image = image.astype(np.uint8)
  # h and w are swapped for landmarks because for images,
  # x and y axes are axis 1 and 0 respectively
  landmarks = landmarks * [new_w / w, new_h / h]
  return image, landmarks


"""Rescale the image in a sample to a given size.

Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
"""

class Rescale(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
    image, landmarks = resize(image, landmarks, self.output_size)
    return {'image': image, 'landmarks': landmarks}


class Rotate(object):
  def __init__(self, chance):
    self.chance = chance
    self.angles = [-30, -15, -10, -5, 5, 10, 15, 30]

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
    
    if random.random() < self.chance:
      angle =  random.sample(self.angles, 1)[0]
      image, landmarks = rotate(image, landmarks, angle)
    
    return {'image': image, 'landmarks': landmarks}


class Hflip(object):
  def __init__(self, chance):
    self.chance = chance

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
    
    if random.random() < self.chance:
      image, landmarks = hflip(image, landmarks)
    
    return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img_shape = image.shape
        
        landmarks = torch.from_numpy(landmarks)
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = TF.to_tensor(image)
        #image = image.transpose((2, 0, 1))
        
        return {'image': image, 'landmarks': landmarks}


class NormTensor(object):

  def __init__(self, norm1 = [0.485, 0.456, 0.406], norm2 = [0.229, 0.224, 0.225]):
    self.norm1 = norm1
    self.std = norm2

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
    
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    landmarks = landmarks - 0.5
    
    return {'image': image, 'landmarks': landmarks}