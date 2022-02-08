import torch
from torchmetrics import F1, Precision, Recall, MatthewsCorrcoef, PSNR, IoU, MeanAbsoluteError, Metric
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import cv2

from utils import load_config

config = load_config("config.yaml")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# NRM
class NegativeRate(Metric):
    def __init__(self, dist_sync_on_step=False):                                
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("TP", default=torch.tensor(0))
        self.add_state("TN", default=torch.tensor(0))
        self.add_state("FP", default=torch.tensor(0))
        self.add_state("FN", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        assert preds.shape == target.shape
        self.TP += preds[target == 1].sum()
        self.TN += -(preds[target == 0] - 1).sum()

        self.FP += -(target[preds == 1] - 1).sum()
        self.FN += target[preds == 0].sum()

    def compute(self):
        NR_fn = self.FN / (self.FN + self.TP)
        NR_fp = self.FP / (self.FP + self.TN)
        return (NR_fn + NR_fp) / 2


#FOM
def fom_fn(y_hat, y, alpha=1.0 / 9):
    """
    Computes Pratt's Figure of Merit for the given image 'y_hat', using a gold
    standard image 'y' as source of the ideal edge pixels.
    """
    alpha = 1.0 / 9
    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).

    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(y))

    fom = 1.0 / np.maximum(
        np.count_nonzero(y_hat),
        np.count_nonzero(y))

    N, M = img.shape

    for i in xrange(0, N):
        for j in xrange(0, M):
            if y_hat[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(y_hat),
        np.count_nonzero(y))    

    return fom


#DRD
def drd_fn(im, im_gt):
    height, width = im.shape
    neg = np.zeros(im.shape)
    neg[im_gt!=im] = 1
    y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
    
    n = 2
    m = n*2+1
    W = np.zeros((m,m), dtype=np.uint8)
    W[n,n] = 1.
    W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    W[n,n] = 1.
    W = 1./W
    W[n,n] = 0.
    W /= W.sum()
    
    nubn = 0.
    block_size = 8
    for y1 in range(0, height, block_size):
        for x1 in range(0, width, block_size):
            y2 = min(y1+block_size-1,height-1)
            x2 = min(x1+block_size-1,width-1)
            block_dim = (x2-x1+1)*(y1-y1+1)
            block = 1-im_gt[y1:y2, x1:x2]
            block_sum = np.sum(block)
            if block_sum > 0 and block_sum < block_dim:
                nubn += 1

    drd_sum= 0.
    tmp = np.zeros(W.shape)
    for i in range(min(1,len(y))):
        tmp[:,:] = 0 

        x1 = max(0, x[i]-n)
        y1 = max(0, y[i]-n)
        x2 = min(width-1, x[i]+n)
        y2 = min(height-1, y[i]+n)

        yy1 = y1-y[i]+n
        yy2 = y2-y[i]+n
        xx1 = x1-x[i]+n
        xx2 = x2-x[i]+n

        tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
        tmp *= W

        drd_sum += np.sum(tmp)
    return drd_sum/(nubn + 10**-8)
 

def DRD(preds, target):
    drd = 0
    for i in range(len(preds)):
        im, im_gt = preds[i].squeeze(0).detach().cpu().numpy(), target[i].squeeze(0).detach().cpu().numpy()
        drd += drd_fn(im, im_gt)
    return drd / len(preds)



def define_metrics(filter_name):
    if filter_name == 'niblack':                                                  # 1 - белый, 0 - черный (pos_label = 0)
        metrics = {'F1':             F1(average=None, num_classes=2)[0],
                  'Precision':       Precision(average=None, num_classes=2)[0], 
                  'Recall':          Recall(average=None, num_classes=2)[0], 
                  'PSNR':            PSNR(), 
                  'NRM':             NegativeRate(),
                  'DRD':             DRD,
                  'IoU':             IoU(num_classes=2)}

    elif filter_name == 'canny':                                    # 1 - край (белый), 0 - фон (черный) (pos_label = 1)
        metrics = {'Matthews Corr':  MatthewsCorrcoef(num_classes=2),
                  'IoU':             IoU(num_classes=2)}
  
    elif filter_name in ['dilation disk', 'dilation square']:
        metrics = {'MAE':  MeanAbsoluteError()}
              
    return metrics


def calculate_metrics(model, test_loader, metrics_dict, metrics_calc):
    '''
    Calculate metrics by all test dataset
    '''

    for metric_name, metric in metrics_dict.items():
        val = 0
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.int()
            probas, y_hat = model(x)
            probas, y_hat = probas.cpu().int(), y_hat.cpu().int()
            if metric_name not in ['DRD']:
                y = y.flatten()
                y_hat = y_hat.flatten()    
            val += metric(y_hat, y).item()      
        metrics_calc[metric_name].append(val/len(test_loader))

        
def define_comparison_table(filter_name):
    metrics = define_metrics(filter_name)
    columns = list(metrics.keys()) + ['Val Loss']
    table = pd.DataFrame(columns=columns)
    return table