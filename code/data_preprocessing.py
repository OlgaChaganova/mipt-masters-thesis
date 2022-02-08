import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color, feature, filters, morphology
import numpy as np
import os
from pathlib import Path
from operator import itemgetter
from tqdm import tqdm
import pyDOE as doe # LHS
from scipy.stats.distributions import uniform

from utils import load_config

config = load_config("config.yaml")


def to_float32(image, float32_ok=False):
    imtype = image.dtype
    if imtype == np.int64 or imtype == np.int32 or imtype == np.uint8 or imtype == np.bool or (imtype == np.float32 and float32_ok):
        image = image.astype(np.float32) / 255
        image = image ** config['GAMMA']
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        image = np.mean(image, axis=2)
    elif imtype == torch.int64 or imtype == torch.int64 or imtype == torch.uint8:
        image = torch.tensor(image, dtype=torch.float32) / 255
        image = image ** config['GAMMA']
        if len(image.shape) < 3:
            image = image.unsqueeze(0)
        image = torch.mean(image, axis=0)   
    return image
    
    
def to_uint8(image):
    imtype = image.dtype
    if imtype == torch.float32 or imtype == torch.float16:
        image = image ** (1/2.2)
        if len(image.size()) == 4:
            axis = 1     
            if image.shape[0] > 1:
                image = image[:, 0:1, :, :] 
        else:
            axis = 0
            if image.shape[0] > 1:
                image = image[0:1, :, :] 
        image = torch.cat([image, image, image], axis=axis)
        image = (image * 255).to(torch.uint8)
    elif imtype == np.float32 or imtype == np.float16:
        image = image ** (1/2.2)
        image = np.concatenate([image, image, image], axis=2)
        image = (image * 255).astype(np.uint8)
    return image    


#source: http://chaladze.com/l5/
# class Linnaeus5(Dataset):
#     def __init__(self, files, filter_name, mode, add_channel=True):
#         super().__init__()
#         self.files = sorted(files)
#         self.len_ = len(self.files)
#         self.filter_name = filter_name
#         self.mode = mode
#         self.add_channel = add_channel
        
#         assert filter_name in ['dilation disk', 'dilation square', 'canny', 'harris', 'niblack'], 'filter_name must be one of the following: dilation disk, dilation square, canny, niblack'
#         if filter_name == 'dilation disk':
#             self.filter_func = lambda x, radius : morphology.dilation(x, selem=morphology.disk(radius=radius))
#         elif filter_name == 'dilation square':
#             self.filter_func = lambda x, width : morphology.dilation(x, selem=morphology.square(width=width))
#         elif filter_name == 'canny':
#             self.filter_func = lambda img, sigma: feature.canny(img, sigma=sigma)
#         elif filter_name == 'niblack':
#             self.filter_func = lambda img, window_size, k: img > filters.threshold_niblack(img, window_size=window_size, k=k)
            
#         self.params = []
#         self.num_params = 0
          
            
#     def __len__(self):
#         return self.len_
     
        
#     def load_sample(self, file):
#         image = Image.open(file)
#         image.load()
#         return image

    
#     def params_init(self, params, num_params):
#         self.params = params
#         self.num_params = num_params
       
        
#     def params_random_init(self, num_params): #LHS; 0.7 sec
#         if self.filter_name in ['dilation disk', 'dilation square']:
#             params = np.random.choice(np.arange(1, 21), size=num_params)# size of the window
#             params = [[params[i]] for i in range(num_params)]
#         elif self.filter_name in ['canny']:
#             params = np.random.uniform(0, 3, size=num_params) # sigma
#             params = [[params[i]] for i in range(num_params)]
#         elif self.filter_name in ['niblack']:
#             win_size = np.random.choice(np.arange(3, 30, 2), size=num_params) 
#             k = np.random.uniform(0, 0.8, size=num_params)
#             params = [[win_size[i], k[i]] for i in range(num_params)]
#         self.params = params
#         self.num_params = num_params
    
    
#     def parameterize_input(self, x, params):  # 7e-4 sec
#         x_shape = x.shape
#         x_param = [x]
#         for i in range(len(params)):
#             x_param.append(torch.full(size=x_shape, fill_value=params[i], dtype=torch.float32))
#         x_param = torch.cat(x_param, dim=0)
#         return x_param

    
#     def __getitem__(self, index):
#         x = self.load_sample(self.files[index])
#         x = np.array(x)
#         x = to_float32(x) # linear processing
        
#         # parametrization: <params_num> sets of parameters per epoch
#         k = np.random.choice(self.num_params)
#         params = self.params[k]
#         if self.mode in ['valid-param', 'test']:
#             print('Params:', *params)
#         y = self.filter_func(x, *params)
         
#         x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
#         if self.add_channel:
#             x = self.parameterize_input(x, params)
#         y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

#         return x, y  


class Linnaeus5(Dataset):
    def __init__(self, files, filter_name, mode, add_channel, is_linear, use_cache=False):
        super().__init__()
        self.files = sorted(files)
        self.len_ = len(self.files)
        self.filter_name = filter_name
        self.mode = mode
        self.use_cache = use_cache
        self.is_linear = is_linear
        self.cached_data = [[], []]
        
        assert filter_name in ['dilation disk', 'dilation square', 'canny', 'harris', 'niblack'], 'filter_name must be one of the following: dilation disk, dilation square, canny, niblack'
        if filter_name == 'dilation disk':
            self.filter_func = lambda x, radius : morphology.dilation(x, selem=morphology.disk(radius=radius))
            self.params = [config['RADIUS']]
            
        elif filter_name == 'dilation square':
            self.filter_func = lambda x, width : morphology.dilation(x, selem=morphology.square(width=width))
            self.params = [config['WIDTH']]
            
        elif filter_name == 'canny':
            self.filter_func = lambda img, sigma: feature.canny(img, sigma=sigma)
            self.params = [config['SIGMA']]
            
        elif filter_name == 'niblack':
            self.filter_func = lambda img, window_size, k: img > filters.threshold_niblack(img, window_size=window_size, k=k)
            self.params = [config['WIN_SIZE'], config['K']]
            
    def __len__(self):
        return self.len_
     
        
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image  
    
    
    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data[0] = torch.stack(self.cached_data[0])
            self.cached_data[1] = torch.stack(self.cached_data[1])
        else:
            self.cached_data = []
        self.use_cache = use_cache
    
    
    def __getitem__(self, index):
        if self.use_cache:
            x, y = self.cached_data[0][index], self.cached_data[1][index]
        else:
            x = self.load_sample(self.files[index])
            x = np.array(x)
            if self.is_linear:
                x = to_float32(x) # linear processing
            else:
                x = color.rgb2gray(x)

            # parametrization: <params_num> sets of parameters per epoch
            y = self.filter_func(x, *self.params)

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            
            self.cached_data[0].append(x)
            self.cached_data[1].append(y)
        return x, y  

    
###
class ParLinnaeus5(Dataset):
    def __init__(self, files, filter_name, mode, add_channel, is_linear):
        super().__init__()
        self.files = sorted(files)
        self.len_ = len(self.files)
        self.filter_name = filter_name
        self.mode = mode
        self.add_channel = add_channel
        self.is_linear = is_linear
        
        assert filter_name in ['dilation disk', 'dilation square', 'canny', 'harris', 'niblack'], 'filter_name must be one of the following: dilation disk, dilation square, canny, niblack'
        if filter_name == 'dilation disk':
            self.filter_func = lambda x, radius : morphology.dilation(x, selem=morphology.disk(radius=radius))
        elif filter_name == 'dilation square':
            self.filter_func = lambda x, width : morphology.dilation(x, selem=morphology.square(width=width))
        elif filter_name == 'canny':
            self.filter_func = lambda img, sigma, low, high: feature.canny(img, sigma=sigma,
                                                                           low_threshold=low, high_threshold=high)
        elif filter_name == 'niblack':
            self.filter_func = lambda img, window_size, k: img > filters.threshold_niblack(img, window_size=window_size, k=k)
            
        self.params = []
        self.num_params = 0
          
            
    def __len__(self):
        return self.len_
     
        
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    
    def params_init(self, params, num_params):
        self.params = params
        self.num_params = num_params
       
        
    def params_random_init(self, num_params): #LHS; 0.7 sec
        if self.filter_name in ['dilation disk', 'dilation square']:
            max_val = 21 if self.filter_name in ['dilation disk'] else 41
            params = np.random.choice(np.arange(1, max_val), size=num_params)# size of the window
            params = [[params[i]] for i in range(num_params)]
            
        elif self.filter_name in ['canny']:
            params = doe.lhs(3, samples=num_params) 
            loc = [0, 0.1, 1.05]
            scale = [3, 0.1, 0.95]
            for j in range(3):
                params[:, j] = uniform(loc=loc[j], scale=scale[j]).ppf(params[:, j])
            params = [[params[i][0], params[i][1], params[i][1]*params[i][2]] for i in range(num_params)]
        
        elif self.filter_name in ['niblack']:
            win_size = np.random.choice(np.arange(3, 30, 2), size=num_params) 
            k = np.random.uniform(0, 0.8, size=num_params)
            params = [[win_size[i], k[i]] for i in range(num_params)]
            
        self.params = params
        self.num_params = num_params
            
              
    def parameterize_input(self, x, params):  # 7e-4 sec
        x_shape = x.shape
        x_param = [x]
        for i in range(len(params)):
            x_param.append(torch.full(size=x_shape, fill_value=params[i], dtype=torch.float32))
        x_param = torch.cat(x_param, dim=0)
        return x_param
    
    
    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = np.array(x)
        if self.is_linear:
            x = to_float32(x) # linear processing
        else:
            x = color.rgb2gray(x)
        
        # parametrization: <params_num> sets of parameters per epoch
        k = np.random.choice(self.num_params)
        params = self.params[k]
        if self.mode in ['valid-param', 'test']:
            print('Params:', *params)
        y = self.filter_func(x, *params)
         
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = self.parameterize_input(x, params)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y

###

class ParLinnaeus5_v2(Dataset):
    def __init__(self, files, filter_name, mode, add_channel, is_linear):
        super().__init__()
        self.files = sorted(files)
        self.len_ = len(self.files)
        self.filter_name = filter_name
        self.mode = mode
        self.add_channel = add_channel
        self.is_linear = is_linear
        
        assert filter_name in ['dilation disk', 'dilation square', 'canny', 'harris', 'niblack'], 'filter_name must be one of the following: dilation disk, dilation square, canny, niblack'
        if filter_name == 'dilation disk':
            self.filter_func = lambda x, radius : morphology.dilation(x, selem=morphology.disk(radius=radius))
        elif filter_name == 'dilation square':
            self.filter_func = lambda x, width : morphology.dilation(x, selem=morphology.square(width=width))
        elif filter_name == 'canny':
            self.filter_func = lambda img, sigma, low, high: feature.canny(img, sigma=sigma,
                                                                           low_threshold=low, high_threshold=high)
        elif filter_name == 'niblack':
            self.filter_func = lambda img, window_size, k: img > filters.threshold_niblack(img, window_size=window_size, k=k)
            
        self.params = []
        self.num_params = 0
          
            
    def __len__(self):
        return self.len_
     
        
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    
    def params_init(self, params, num_params):
        self.params = params
        self.num_params = num_params
       
        
    def params_random_init(self, num_params): #LHS; 0.7 sec
        if self.filter_name in ['dilation disk', 'dilation square']:
            max_val = 21 if self.filter_name in ['dilation disk'] else 41
            params = np.random.choice(np.arange(1, max_val), size=num_params)# size of the window
            params = [[params[i]] for i in range(num_params)]
            
        elif self.filter_name in ['canny']:
            params = doe.lhs(3, samples=num_params) 
            loc = [0, 0.1, 1.05]
            scale = [3, 0.1, 0.95]
            for j in range(3):
                params[:, j] = uniform(loc=loc[j], scale=scale[j]).ppf(params[:, j])
            params = [[params[i][0], params[i][1], params[i][1]*params[i][2]] for i in range(num_params)]
        
        elif self.filter_name in ['niblack']:
            win_size = np.random.choice(np.arange(3, 30, 2), size=num_params) 
            k = np.random.uniform(0, 0.8, size=num_params)
            params = [[win_size[i], k[i]] for i in range(num_params)]
            
        self.params = params
        self.num_params = num_params
            
              
    def parameterize_input(self, x, params):  # 7e-4 sec
        x_shape = x.shape
        x_param = [x]
        for i in range(len(params)):
            x_param.append(torch.full(size=x_shape, fill_value=params[i], dtype=torch.float32))
        x_param = torch.cat(x_param, dim=0)
        return x_param
        
        
    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = np.array(x)
        if self.is_linear:
            x = to_float32(x) # linear processing
        else:
            x = color.rgb2gray(x)
        
        # parametrization: <params_num> sets of parameters per epoch
        k = np.random.choice(self.num_params)
        params = self.params[k]
        if self.mode in ['valid-param', 'test']:
            print('Params:', *params)
        y = self.filter_func(x, *params)
         
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        params = torch.tensor(params, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return (x, params), y

###
   
def get_data(data_dir, is_silent=False):
    train_dir = Path(os.path.join(data_dir, 'train'))
    test_dir = Path(os.path.join(data_dir, 'test'))

    train_files = list(train_dir.rglob('*.jpg'))
    test_files = list(test_dir.rglob('*.jpg'))

    ids = np.random.permutation(len(test_files))
    N = len(test_files) // 2

    valid_files = list(itemgetter(*ids[:N])(test_files))
    test_files = list(itemgetter(*ids[N:])(test_files))
    
    if not is_silent:
        print('Files are loaded.')
        print('Train size: ', len(train_files), '\t', 'Valid size: ', len(valid_files), '\t', 'Test size: ', len(test_files))
    return train_files, valid_files, test_files    


def get_dataloaders(dataset, filter_name, add_channel, is_parameterized, is_linear):
    train_files, valid_files, test_files = get_data(config['DATA_DIR'])
    
    train_dataset = dataset(train_files, filter_name, mode='train', add_channel=add_channel, is_linear=is_linear)
    valid_dataset = dataset(valid_files, filter_name, mode='valid', add_channel=add_channel, is_linear=is_linear)
    valid_param_dataset = dataset(valid_files, filter_name, mode='valid-param', add_channel=add_channel, is_linear=is_linear)
    test_dataset =  dataset(test_files,  filter_name, mode='test', add_channel=add_channel, is_linear=is_linear) 
    
    if is_parameterized:
        train_dataset.params_random_init(config['NUM_SET_PARAMS']) # создаем <NUM_SET_PARAMS> рандомных наборов параметров
        valid_dataset.params_init(train_dataset.params, config['NUM_SET_PARAMS'])
        valid_param_dataset.params_init(train_dataset.params, config['NUM_SET_PARAMS'])
        test_dataset.params_init(train_dataset.params, config['NUM_SET_PARAMS'])
              
    train_loader = DataLoader(train_dataset, config['BATCH_SIZE'], drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_dataset, config['BATCH_SIZE'], drop_last=False, shuffle=True)
    valid_param_loader = DataLoader(valid_param_dataset, config['BATCH_SIZE'], drop_last=False, shuffle=True)
    test_loader  = DataLoader(test_dataset, config['BATCH_SIZE'], drop_last=False, shuffle=True)

    return train_loader, valid_loader, valid_param_loader, test_loader


def cache_dataloader(dataloader):
    dataloader.num_workers = 0
    print('Caching the data...')
    for data in tqdm(dataloader):
        True
    dataloader.dataset.set_use_cache(use_cache=True)
    dataloader.num_workers = 2
    return dataloader
