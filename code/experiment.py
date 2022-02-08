import torch
import matplotlib.pyplot as plt
from torchsummary import summary
import importlib

from data_preprocessing import get_dataloaders, cache_dataloader
from torch_receptive_field import receptive_field
from training import train, plot_loss, plot_metrics, plot_ROC_curve
from metrics import define_metrics, calculate_metrics
from visualization import visualize_result#, visualize_lena
from utils import save_trained_model, load_config


config = load_config("config.yaml")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_experiment(filter_name,
                   model, optimizer, criterion, scheduler, stopping, dataset, is_parameterized, use_cache, is_linear,
                   model_name, add_channel, comp_table, writer, start_idx=0):
    
    _, _, _, test_loader = get_dataloaders(dataset, filter_name, add_channel, is_parameterized, is_linear)
    
    if use_cache:
        test_loader = cache_dataloader(test_loader)
        
    if is_parameterized:
        num_channels = 2 if filter_name in ['dilation disk', 'dilation square'] else 4
    else:
        num_channels = 1
        
    input_shape = (num_channels, config['IMG_SIZE'], config['IMG_SIZE'])

    model = model.to(DEVICE)
    print('\n************    Starting experiment    ************\n')
    print('Model:', model)

    print('\nModel\'s summary:')
    try:
        print(summary(model, input_shape))
    except:
        print('Model\'s summary is not available')

    print('\nTraining:')

    trained_model, history, metrics_calc = train(model, optimizer, dataset, scheduler, stopping,
                                                 criterion, filter_name, model_name, add_channel,
                                                 is_parameterized, use_cache, is_linear, start_idx)

    # show training and metric curves 
    fig, axs = plt.subplots(1, 2, figsize=(18,5))
    plot_loss(axs[0], history)
    plot_metrics(axs[1], metrics_calc)    
    plt.show()

#     # ROC curve
#     if filter_name in ['canny', 'niblack']:
#         fig, axs = plt.subplots(1, 1, figsize=(9,5))
#         plot_ROC_curve(axs, model, test_loader)
#         plt.show()

    #show metrics
    print('\n\nMetrics on test dataset:')
    try:
        metrics = define_metrics(filter_name)
        metrics_calc_test = {key: [] for key in metrics.keys()}
        calculate_metrics(trained_model, test_loader, metrics, metrics_calc_test)
        print(metrics_calc_test)
    except:
        print('Metrics cannot be calculated')

#     # add visualization
#     print('\n\n\t\t\t\t\t\t\t Visualization on test dataset')
#     visualize_result(model, filter_name, test_loader, criterion, is_silent=False)
#     print ('\n\n\t\t\t\t\t\t\t Lena')
#     visualize_lena(model, filter_func, filter_name)

    # add metric to comparison table
    if comp_table is not None:
#         comp_table.loc[model_name, :] = [item[-1] for item in list(metrics_calc_test.values())] + [history['valid loss by epoch'][-1]]
        comp_table.loc[model_name, :] = [item for sublist in list(metrics_calc_test.values()) for item in sublist] + [history['valid loss by epoch'][-1]]
            

    return trained_model, history, metrics_calc, writer