from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *

import scipy.io as sio


def count_size_images_mpi(listOfFiles):
  shapes = [0, 0, 0]
  for i, file_name in enumerate(listOfFiles):
    mat = sio.loadmat(listOfFiles[i])
    for side in ['left', 'right']:
      shapes[0] += mat['data'][side][0][0]['image'][0][0].shape[0]
  shapes[1] = mat['data'][side][0][0]['image'][0][0].shape[1]
  shapes[2] = mat['data'][side][0][0]['image'][0][0].shape[2]
  return shapes, mat['data'][side][0][0]['image'][0][0].dtype


def read_files_mpi_val(eval_files, path):
  full_names = []
  for ef in eval_files:
    person = os.path.splitext(os.path.basename(ef))[0]
    with open(ef) as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if line != '':
          img_path, side = [x.strip() for x in line.split()]
          day, img = img_path.split('/')

          full_name = os.path.join(path, person, day, side, img)
          full_names.append(full_name)
  return full_names


def process_mpi_files(listOfFiles):
  images_index_name = {}

  images = np.zeros((427316, 36, 60), dtype = np.uint8)
  gazes = np.zeros((427316, 2))

  real_index = 0

  for i, file_name in enumerate(listOfFiles):
    mat = sio.loadmat(listOfFiles[i])
    for side in ['left', 'right']:
      batch_images = mat['data'][side][0][0]['image'][0][0]
      images_n = batch_images.shape[0]

      if side == 'right':
        batch_images = np.fliplr(batch_images)

      images[real_index:real_index+images_n] = batch_images
      
      batch_gazes = mat['data'][side][0, 0]['gaze'][0, 0]

      theta = np.arcsin(-batch_gazes[:, 1])
      phi = np.arctan2(-batch_gazes[:, 0], -batch_gazes[:, 2])

      gazes[real_index:real_index+images_n, 0] = -theta
      gazes[real_index:real_index+images_n, 1] = phi

      if side == 'right':
        gazes[:, 1] = -gazes[:, 1]

      for indexing_image in range(images_n):
        img_name = mat['filenames'][indexing_image][0][0]
        img_name = os.path.join(listOfFiles[i][:-4], side, img_name)
        images_index_name[img_name] = real_index + indexing_image

      real_index += images_n

  return images, gazes, images_index_name


def devide_val(images, gazes, images_index_name, val_list):

  indexes_val = []
  for filename in val_list:
    if images_index_name[filename]:
      indexes_val.append(images_index_name[filename])
  
  images_train = np.delete(images, indexes_val, axis=0)
  gazes_train = np.delete(gazes, indexes_val, axis=0)

  images_val = images[indexes_val]
  gazes_val = gazes[indexes_val]

  return images_train, gazes_train, images_val, gazes_val



def pitchyaw_to_vector(pitchyaws):
  """
  Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
  Args:
      pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
  Returns:
      :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
  """
  n = pitchyaws.shape[0]
  sin = torch.sin(pitchyaws)
  cos = torch.cos(pitchyaws)
  out = torch.empty((n, 3))
  out[:, 0] = torch.mul(cos[:, 0], sin[:, 1])
  out[:, 1] = sin[:, 0]
  out[:, 2] = torch.mul(cos[:, 0], cos[:, 1])
  return out


radians_to_degrees = 180.0 / np.pi


def angularError(a, b):
  """Calculate angular error (via cosine similarity)."""
  a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
  b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

  ab = torch.sum(torch.mul(a, b), dim=1)
  a_norm = torch.linalg.norm(a, dim=1)
  b_norm = torch.linalg.norm(b, dim=1)

  # Avoid zero-values (to avoid NaNs)
  a_norm = torch.clamp(a_norm, min=1e-7,)
  b_norm = torch.clamp(b_norm, min=1e-7,)

  similarity = torch.div(ab, torch.mul(a_norm, b_norm))

  return torch.acos(similarity) * radians_to_degrees