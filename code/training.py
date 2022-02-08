import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from torchmetrics import ROC, AUROC
import gc

from utils import load_config, save_trained_model
from metrics import define_metrics, calculate_metrics
from visualization import visualize_result
from data_preprocessing import get_dataloaders, cache_dataloader

config = load_config("config.yaml")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(model, opt, dataset, scheduler, stopping, criterion, filter_name, model_name,
          add_channel, is_parameterized, use_cache, is_linear, start_idx=0):
    
  train_loader, valid_loader, valid_param_loader, test_loader = get_dataloaders(dataset, 
                                                                                filter_name, 
                                                                                add_channel, 
                                                                                is_parameterized,
                                                                                is_linear)
  if use_cache:
        train_loader = cache_dataloader(train_loader)
        valid_loader = cache_dataloader(valid_loader)
        valid_param_loader = cache_dataloader(valid_param_loader)
        test_loader = cache_dataloader(test_loader)
        

  metrics = define_metrics(filter_name)
  metrics_calc = {key: [] for key in metrics.keys()}

  history = {'train loss by epoch': [],
             'valid loss by epoch': [],
             'train loss by batch': [],
             'valid loss by batch': []}

  log_template = "Epoch {ep:d}:\t mean train_loss: {t_loss:0.6f}\t mean val_loss {v_loss:0.6f}\n"
  iters = len(train_loader)
  best_loss_val = 10*6

  for epoch in range(config['EPOCHS']):
      model.train()
      train_loss_per_epoch = []
      val_loss_per_epoch = []
      i = 0
      for x, y in tqdm(train_loader):
            
          try:
              x, y = x.to(DEVICE), y.to(DEVICE)
          except:
              x = [_x.to(DEVICE) for _x in x] 
              y = y.to(DEVICE)
            
          opt.zero_grad()
        
          try:
              probas, y_hat = model(x)
          except:
              probas, y_hat = model(*x)
                
          loss = criterion(probas, y)
          train_loss_per_epoch.append(loss.item())
          loss.backward()
          opt.step()

          if scheduler:
            scheduler.step(epoch + i / iters)
            i += 1

      history['train loss by epoch'].append(np.mean(train_loss_per_epoch))
      history['train loss by batch'].extend(train_loss_per_epoch)

      model.eval()
      with torch.no_grad():
          for x, y in valid_loader:
              try:
                  x, y = x.to(DEVICE), y.to(DEVICE)
              except:
                  x = [_x.to(DEVICE) for _x in x] 
              y = y.to(DEVICE)

              try:
                  probas, y_hat = model(x)
              except:
                  probas, y_hat = model(*x)
                    
              loss = criterion(probas, y)
              val_loss_per_epoch.append(loss.item())

          history['valid loss by epoch'].append(np.mean(val_loss_per_epoch))
          history['valid loss by batch'].extend(val_loss_per_epoch)

      try:
        calculate_metrics(model, valid_loader, metrics, metrics_calc)
      except:
        print('Metrics cannot be calculated')


      tqdm.write(log_template.format(ep=epoch+1, t_loss=np.mean(train_loss_per_epoch),
                                      v_loss=np.mean(val_loss_per_epoch)))
        
      if stopping:
          stopping(np.mean(val_loss_per_epoch)) # val_loss_per_epoch[epoch]
          if stopping.early_stop:
              break
                
      if start_idx + epoch > 5 and best_loss_val > np.mean(val_loss_per_epoch):
        print('The best loss on validation is reached. Saving the model.')
        best_loss_val = np.mean(val_loss_per_epoch)
        save_trained_model(model, opt, criterion, epoch+1, model_name, start_idx=start_idx)
        visualize_result(model, valid_param_loader, filter_name, criterion, is_linear, is_silent=False)
        print('\n')
      
      gc.collect()     
        
  return model, history, metrics_calc




# def train(model, opt, dataset, scheduler, stopping, criterion, filter_name, model_name, add_channel, start_idx=0):
#   train_loader, valid_loader, valid_param_loader, test_loader = get_dataloaders(dataset, filter_name, add_channel)

#   metrics = define_metrics(filter_name)
#   metrics_calc = {key: [] for key in metrics.keys()}

#   history = {'train loss by epoch': [],
#              'valid loss by epoch': [],
#              'train loss by batch': [],
#              'valid loss by batch': []}

#   log_template = "Epoch {ep:d}:\t mean train_loss: {t_loss:0.4f}\t mean val_loss {v_loss:0.4f}\n"
#   iters = len(train_loader)
#   best_loss_val = 10*6

#   for epoch in range(config['EPOCHS']):
#       model.train()
#       train_loss_per_epoch = []
#       val_loss_per_epoch = []
#       i = 0
#       for x, y in tqdm(train_loader):
#           if model.__class__.__name__ == 'ParamNet':
#               x, params = x
#               x, params, y = x.to(DEVICE), params.to(DEVICE), y.to(DEVICE)
#               opt.zero_grad()
#               probas, y_hat = model(x, params)
#           elif model.__class__.__name__ == 'ConvNet':
#               x, y = x.to(DEVICE), y.to(DEVICE)
#               opt.zero_grad()
#               probas, y_hat = model(x)
#           loss = criterion(probas, y)
#           train_loss_per_epoch.append(loss.item())
#           loss.backward()
#           opt.step()

#           if scheduler:
#             scheduler.step(epoch + i / iters)
#             i += 1

#       history['train loss by epoch'].append(np.mean(train_loss_per_epoch))
#       history['train loss by batch'].extend(train_loss_per_epoch)

#       model.eval()
#       with torch.no_grad():
#           for x, y in valid_loader:
#               if model.__class__.__name__ == 'ParamNet':
#                   x, params = x
#                   x, params, y = x.to(DEVICE), params.to(DEVICE), y.to(DEVICE)
#                   probas, y_hat = model(x, params)
#               elif model.__class__.__name__ == 'ConvNet':
#                   x, y = x.to(DEVICE), y.to(DEVICE)
#                   opt.zero_grad()
#                   probas, y_hat = model(x)
#               loss = criterion(probas, y)
#               val_loss_per_epoch.append(loss.item())

#           history['valid loss by epoch'].append(np.mean(val_loss_per_epoch))
#           history['valid loss by batch'].extend(val_loss_per_epoch)

#       try:
#         calculate_metrics(model, valid_loader, metrics, metrics_calc)
#       except:
#         print('Metrics cannot be calculated')


#       tqdm.write(log_template.format(ep=epoch+1, t_loss=np.mean(train_loss_per_epoch),
#                                       v_loss=np.mean(val_loss_per_epoch)))

#       if epoch % 5 == 0:
#         visualize_result(model, valid_param_loader, filter_name, criterion, is_silent=False)
#         print('\n')
        
#       if stopping:
#           stopping(np.mean(val_loss_per_epoch)) # val_loss_per_epoch[epoch]
#           if stopping.early_stop:
#               break
                
#       if epoch > 5 and best_loss_val > np.mean(val_loss_per_epoch):
#         print('The best loss on validation is reached. Saving the model.')
#         best_loss_val = np.mean(val_loss_per_epoch)
#         save_trained_model(model, opt, criterion, epoch+1, model_name, start_idx=start_idx)
#         visualize_result(model, valid_param_loader, filter_name, criterion, is_silent=False)
#         print('\n')
        
#   return model, history, metrics_calc


# 2 graphics
def plot_loss(ax, history):
  ax.plot(history['train loss by epoch'], label='train')
  ax.plot(history['valid loss by epoch'], label='val')
  ax.set_xlabel("Epoch")
  ax.set_xticks(np.arange(0, config['EPOCHS'], 2))
  ax.set_ylabel("Loss")
  ax.set_title('Loss Curve')
  ax.legend()


def plot_metrics(ax, metrics_calc):
  for m_name, m_values in metrics_calc.items():
    if m_name == 'PSNR':
      ax2=ax.twinx()
      ax2.plot(m_values, 'c--', label=m_name)
      ax2.legend(loc='upper right')
    else:
      ax.plot(m_values, label=m_name)

  ax.set_xlabel("Epoch")
  ax.set_xticks(np.arange(0, config['EPOCHS'], 2))
  ax.set_ylabel("Metrics value")
  ax.legend(loc='upper left')
  ax.set_title('Metric Curves')


def plot_ROC_curve(ax, model, test_loader):
  roc = ROC(pos_label=1)
  auroc = AUROC(pos_label=1)

  x_rgb, x_gray, y = next(iter(test_loader))
  x = x_rgb.to(DEVICE) if config['IS_RGB'] else x_gray.to(DEVICE)

  _, y_hat = model(x)
  y_hat = y_hat.cpu()

  fpr, tpr, thresholds = roc(y_hat, y)
  auc = auroc(y_hat.flatten(), y.flatten().int()).item()
  ax.plot(fpr, tpr)
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  ax.text(x=0.8, y=0.8, s=str('AUC = %.4f' % auc), bbox=props)
  ax.set_title('ROC Curve')
    