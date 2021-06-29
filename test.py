"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from models.fewshot import FewShotSegNet
from dataset.voc import voc_fewshot
from dataset.transforms import ToTensorNormalize, Resize
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
import config

def test():
    # reproducibility
    set_seed(config.seed)

    # create model
    print('##### Create Model #####')
    model = FewShotSegNet(pretrained_path=config.path['init_path']).cuda()
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
                label_ids = list(sample_batch['class_ids'])
                support_images = [[shot.cuda() for shot in way] for way in sample_batch['support_images']]

                suffix = 'scribble' if config.scribble else 'mask'

                if config.bbox:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batch['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batch['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batch['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batch['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batch['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batch['query_labels']], dim=0)

                query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
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