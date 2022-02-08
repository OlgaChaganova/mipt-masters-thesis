import matplotlib.pyplot as plt
from skimage import color
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from utils import load_config
from data_preprocessing import get_data, to_uint8, to_float32

config = load_config("config.yaml")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_PIL(img_batch, nrow=config['BATCH_SIZE'], inverse=True):
  if inverse:
      img_batch = to_uint8(img_batch)
  to_PIL = transforms.ToPILImage() 
  pil_img_batch = to_PIL(make_grid(img_batch, nrow=nrow))
  display(pil_img_batch)


def imshow(img_batch, title='', nrows=config['BATCH_SIZE'], inverse=True):
    if inverse:
      img_batch = to_uint8(img_batch)
    
    img_batch = make_grid(img_batch, nrows=nrows)
    npimg = img_batch.detach().cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(19, 19))
    plt.axis('off')
    plt.imshow(npimg)
    plt.title(title)
    plt.show()


def visualize_result(model, dataloader, filter_name, criterion, is_linear, type='imshow', use_cuda=True, is_silent=True):
  x, y = next(iter(dataloader))
  if use_cuda:
    x = x.cuda()
  else:
    model = model.cpu()
  probas, y_hat = model(x)

  if x.shape[1] > 1:
    x = x[:, 0:1, :, :]

  x = x.cpu()
  if filter_name in ['canny', 'niblack']:
      y_hat = y_hat.cpu()
  elif filter_name in ['dilation disk', 'dilation square']:
      y_hat = probas.cpu()

  if type == 'imshow':
      imshow(x, 'Original images', inverse=is_linear)
      imshow(y, 'Filtered images', inverse=is_linear)
      imshow(y_hat, 'Network output', inverse=is_linear)
    
  elif type == 'PIL':
      show_PIL(x, inverse=is_linear)
      show_PIL(y, inverse=is_linear)
      show_PIL(y_hat, inverse=is_linear)

  if not is_silent:
      losses = []
      for i in range(y.shape[0]):
          losses.append(round(criterion(y_hat[i], y[i]).item(), 4))
      print('Losses per image:', losses)   