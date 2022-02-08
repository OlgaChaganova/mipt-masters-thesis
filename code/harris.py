import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import feature
from utils import load_config


config = load_config("config.yaml")


def show_corners(x, y, is_rgb, is_predicted, color='yellow'):
  """
  Visualise image with ground truth corners
  - x - original image
  - y - corners (model heatmap / harris heatmap)
  - color - color of corners' markers
  - is_rgb (bool)
  - is_predicted (bool): true - use probability heatmap (y), false - use ground truth map 
  """
  coords = [] 
  title = 'Predicted corners' if is_predicted else 'Corners by Harris detector'
  y_gt = get_corners(y.detach().cpu()) # binary GT 
  for i in range(y_gt.shape[0]):
      coord_y = (y_gt[i].squeeze(0) == 1).nonzero(as_tuple=False).numpy()
      coords.append(coord_y)
    
  fig = plt.figure(1,(19, 5))
  grid = ImageGrid(fig, 111, nrows_ncols=(1, config['BATCH_SIZE']), axes_pad=0)
  
  for i in range(config['BATCH_SIZE']):
      if is_rgb:
        image = x[i,:].permute(1, 2, 0)
        grid[i].imshow(image, interpolation='none')
      else:
        image = x[i,:].squeeze()
        grid[i].imshow(image,cmap='gray',interpolation='none')

      grid[i].plot(coords[i][:, 1], coords[i][:, 0], color=color, marker='*',
            linestyle='None', markersize=7)
      grid[i].plot(coords[i][:, 1], coords[i][:, 0], '.r', markersize=2)
      grid[i].axis('off')

  fig.suptitle(title, y=0.75)
  plt.show()
    
    
def get_corners(y_heatmap):
  tensor_batch = y_heatmap.squeeze(1)
  gt = []

  for i in range(tensor_batch.shape[0]):
    coords = feature.corner_peaks(tensor_batch[i].numpy(),
                                  min_distance=config['MIN_DISTANCE'], threshold_rel=config['THRESHOLD_REL'])
    y_gt = np.zeros_like(tensor_batch[i])
    for x, y in coords:
      y_gt[x, y] = 1
    y_gt = transforms.ToTensor()(y_gt)
    gt.append(y_gt)

  gt = torch.stack(tuple(gt))
  return gt