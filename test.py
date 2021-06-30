"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from models.fewshot import FewShotSegNet
from models.metric import MetricSegNet
from dataset.voc import voc_fewshot
from dataset.transforms import ToTensorNormalize, Resize
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, knn_predict
import config

def test():
    # sanity check
    assert config.mode_type in ['fewshot', 'metric'], f'Unknown model type {config.mode_type}'
    # reproducibility
    set_seed(config.seed)

    # create model
    print('##### Create Model #####')
    if config.mode_type == 'fewshot':
        model = FewShotSegNet(pretrained_path=config.path['init_path']).cuda()
    elif config.mode_type == 'metric':
        model = MetricSegNet(pretrained_path=config.path['init_path']).cuda()
    if not config.notrain:
        model.load_state_dict(torch.load(config.snapshot))
    model.eval()

    # prepare data
    print('##### Prepare data #####')
    labels = CLASS_LABELS['VOC']['all'] - CLASS_LABELS['VOC'][config.label_sets]
    transforms = Compose([
        Resize(size=config.input_size)
    ])

    print('##### Testing Begins #####')
    metric = Metric(max_label=20, n_runs=config.n_runs)
    with torch.no_grad():
        for run in range(config.n_runs):
            print(f'### Run {run + 1} ###')
            set_seed(config.seed + run)

            print(f'### Load data ###')
            dataset = voc_fewshot(
                base_dir=config.path['data_dir'],
                split=config.path['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=config.n_steps*config.batch_size,
                n_ways=config.task['n_ways'],
                n_shots=config.task['n_shots'],
                n_queries=config.task['n_queries']
            )
            testloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, pin_memory=True)

            print(f'Total num of Data: {len(dataset)}')

            for sample_batch in tqdm.tqdm(testloader):
                label_ids = list(sample_batch['class_ids']) # ways
                support_images = [[shot.cuda() for shot in way] for way in sample_batch['support_images']] # ways x shots x [B, 3, H, W]
                
                query_images = [query_image.cuda()
                                for query_image in sample_batch['query_images']] # queries x [B, 3, H, W]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batch['query_labels']], dim=0) # [B*queries, H, W]

                if config.mode_type == 'fewshot':
                    support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                                        for way in sample_batch['support_mask']]
                    support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                                        for way in sample_batch['support_mask']]

                    query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                        query_images)

                    metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                                np.array(query_labels[0].cpu()),
                                labels=label_ids, n_run=run)
                elif config.mode_type == 'metric':
                    # TODO
                    support_labels = torch.cat([torch.cat(way, dim=0) for way in sample_batch['support_labels']]).long().cuda() # [Bxwaysxshots, H, W]
                    H, W = support_labels.shape[-2:]
                    support_fts, query_fts = model(support_images, query_images) # [ways*shots*B, C, Hf, Wf], [queries*B, C, Hf, Wf]
                    Hf, Wf = support_fts.shape[-2:]
                    # downsample support_labels
                    support_labels = F.interpolate(support_labels.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').long().flatten() # [B*ways*shots*Hf*Wf]
                    # reshape support_fts and query_fts
                    support_fts = support_fts.view(-1, support_fts.shape[1]).contiguous() # [B*ways*shots*Hf*Wf, C]
                    query_fts = query_fts.view(-1, query_fts.shape[1]).contiguous() # [B*queries*Hf*Wf, C]
                    # filter out unknown support labels
                    support_ind = (support_labels != config.ignore_label)
                    support_labels, support_fts = support_labels[support_ind], support_fts[support_ind]
                    # use knn to do prediction
                    query_pred = knn_predict(query_fts, support_fts, support_labels, classes=21, knn_k=5).view(-1, Hf, Wf) # [B*queries, Hf, Wf]
                    query_pred = F.interpolate(query_pred.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=True).squeeze(1).long() # [B*queries, H, W]
                    query_pred[query_pred != 0] = 1 # only works for ways = 1
                    metric.record(np.array(query_pred[0].cpu()),
                                np.array(query_labels[0].cpu()),
                                labels=label_ids, n_run=run)
            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)
            print(f'classIoU: {classIoU}')
            print(f'meanIoU: {meanIoU}')
            print(f'classIoU_binary: {classIoU_binary}')
            print(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    print('----- Final Result -----')
    print(f'classIoU mean: {classIoU}')
    print(f'classIoU std: {classIoU_std}')
    print(f'meanIoU mean: {meanIoU}')
    print(f'meanIoU std: {meanIoU_std}')
    print(f'classIoU_binary mean: {classIoU_binary}')
    print(f'classIoU_binary std: {classIoU_std_binary}')
    print(f'meanIoU_binary mean: {meanIoU_binary}')
    print(f'meanIoU_binary std: {meanIoU_std_binary}')


if __name__ == '__main__':
    test()