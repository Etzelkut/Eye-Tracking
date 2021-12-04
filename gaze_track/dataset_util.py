from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *

import json
import cv2

def process_coords(coords_list, ih):
  coords = [eval(l) for l in coords_list]
  return np.array([(x, ih-y, z) for (x, y, z) in coords])


def vector_to_pitchyaw(vectors):
  r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
  Args:
      vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
  Returns:
      :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
  """
  n = vectors.shape[0]
  out = np.empty((n, 2))
  vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
  out[:, 0] = np.arcsin(vectors[:, 1])  # theta
  out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
  return out



def gaussian_2d(w, h, cx, cy, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
    return np.array(heatmaps)


def preprocess_unityeyes_image(img, json_data):

  ih, iw = img.shape[:2]
  #ih_2, iw_2 = ih/2.0, iw/2.0

  interior_landmarks = process_coords(json_data['interior_margin_2d'], ih)
  caruncle_landmarks = process_coords(json_data['caruncle_2d'], ih)
  iris_landmarks = process_coords(json_data['iris_2d'], ih)

  left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
  right_corner = interior_landmarks[8, :2]
  eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
  eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                      np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

  aa = eye_middle[1] - eye_width/3
  aa2 = eye_middle[1] + eye_width/3
  bb = eye_middle[0] - eye_width/2
  bb2 = eye_middle[0] + eye_width/2

  img = img[int(aa):int(aa2), int(bb):int(bb2)]

  look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))
  gaze = vector_to_pitchyaw(-look_vec).flatten()
  gaze = gaze.astype(np.float32)

  iris_center = np.mean(iris_landmarks, axis=0)[:2]

  landmarks = np.concatenate([#interior_landmarks[:, :2],  # 8
                            iris_landmarks[::2, :2],  # 8
                            iris_center.reshape((1, 2)),
                            #[[iw_2, ih_2]],  # Eyeball center
                            ])  # 18 in total
  
  landmarks = landmarks - [eye_middle[0], eye_middle[1]] + [eye_width/2, eye_width/3]
  
  landmarks[:, [0, 1]] = landmarks[:, [1, 0]]

  #heatmaps = get_heatmaps(img.shape[0], img.shape[1], landmarks)
  
  return {
        'img': img,
        #'eye_middle': eye_middle,
        #'heatmaps': heatmaps,
        'landmarks': landmarks,
        'gaze': gaze,
        'look_vec': look_vec,
    }
