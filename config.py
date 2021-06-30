import os
import re

"""Default configurations"""
input_size = (417, 417)
seed = 1234
mode = 'train' # 'train' or 'test'

if mode == 'train':
    n_steps = 30000
    label_sets = 0
    batch_size = 1
    lr_milestones = [10000, 20000, 30000]
    ignore_label = 255
    print_interval = 100
    save_pred_every = 5000
    n_samples = 120 # number of samples in each batch for metric learning
    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
    }

    optim = {
        'lr': 1e-2,
        'momentum': 0.9,
        'weight_decay': 0.0001,
    }

elif mode == 'test':
    notrain = False
    snapshot = f'./runs/1_ways_1_shots/checkpoints/10000.pth'
    n_runs = 5
    n_steps = 1000
    batch_size = 1
    ignore_label = 255

    # Set label_sets from the snapshot string
    label_sets = 0

    # Set task config from the snapshot string
    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
    }
else:
    raise ValueError('Wrong configuration for "mode" !')


path = {
    'log_dir': './runs',
    'init_path': './pretrained_model/vgg16-397923af.pth',
    'data_dir': '../data/VOCdevkit/VOC2012/',
    'data_split': 'trainaug',
}