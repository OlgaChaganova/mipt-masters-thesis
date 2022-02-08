import torch
import pandas as pd
import os
import yaml


def load_config(config_name):
    with open(os.path.join('./', config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config('config.yaml')


def save_trained_model(model, opt, loss, epoch, model_name, start_idx=0):
    try:
        torch.save({
            'state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
        }, config['MODELS_DIR'] + str(model_name) + "-" + str(start_idx + epoch))
        print('Model is DataParallel object.')
        
    except AttributeError:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
        }, config['MODELS_DIR'] + str(model_name) + "-" + str(start_idx + epoch))
        print('Model is not a DataParallel object.')
        
    name = str(model_name) + "-" + str(start_idx + epoch)
    print('\nModel\'s weights were saved. Name: ', name, '\n')
    
    
def load_trained_model(model, optimizer, loss, name):            
    path = config['MODELS_DIR'] + name
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    return model, optimizer, loss
