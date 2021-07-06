from models.loss import ContrasiveLoss
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose

from torch.utils.tensorboard import SummaryWriter

from models.metric import MetricSegNet
from models.fewshot import FewShotSegNet
from models.loss import ContrasiveLoss
from dataset.voc import voc_fewshot
from dataset.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import config


def train():
    # reproducibility
    set_seed(config.seed)
    # sanity check
    assert config.mode_type in ['fewshot', 'metric'], f'Unknown mode type: {config.mode_type}, expect [fewshot, metric]'
    # create model
    print('##### Create Model #####')
    if config.mode_type == 'fewshot':
        model = FewShotSegNet(pretrained_path=config.path['init_path']).cuda()
    else:
        model = MetricSegNet(pretrained_path=config.path['init_path']).cuda()
    model.train()

    # prepare data
    print('##### Prepare data #####')
    labels = CLASS_LABELS['VOC'][config.label_sets]
    transforms = Compose([
        Resize(size=config.input_size),
        RandomMirror()
    ])

    n_ways, n_shots, n_queries = config.task['n_ways'], config.task['n_shots'], config.task['n_queries']
    assert n_ways == 1, 'currently, the code only supports n_ways = 1'
    dataset = voc_fewshot(
        base_dir=config.path['data_dir'],
        split=config.path['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=config.n_steps * config.batch_size,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=n_queries
    )
    print(f'Num of data: {len(dataset)}')
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )

    # set optimizer
    print('##### Set optimizer #####')
    optimizer = torch.optim.SGD(model.parameters(), **config.optim)
    scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)
    if config.mode_type == 'fewshot':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label).cuda() 
    else:
        criterion = ContrasiveLoss().cuda()

    # set logger
    logdir = os.path.join(config.path['log_dir'], f'{n_ways}_ways_{n_shots}_shots', config.mode_type)
    logger = SummaryWriter(log_dir=logdir)
    save_path = f'{logdir}/checkpoints'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # start training
    i_iter = 0
    log_loss = {'loss': 0.0}
    print('##### Training #####')
    for i_iter, sample_batch in enumerate(trainloader):
         # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batch['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batch['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batch['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batch['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batch['query_labels']], dim=0)

        # Forward and Backward
        if config.mode_type == 'fewshot':
            query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                        query_images)
            loss = criterion(query_pred, query_labels)
        else:
            support_fts, query_fts = model(support_images, query_images)
            support_fg_mask = torch.cat([torch.cat(way, dim=0) for way in support_fg_mask], dim=0) # [waysxshotsxB, H, W]
            loss = criterion(torch.cat((support_fts, query_fts), dim=0), torch.cat((support_fg_mask, query_labels), dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        loss = loss.item()
        logger.add_scalar('loss', loss, global_step=i_iter)
        log_loss['loss'] += loss

        # print loss and take snapshots
        if (i_iter + 1) % config.print_interval == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            print(f'step {i_iter + 1}/{len(trainloader)}: loss: {loss}')

        if (i_iter + 1) % config.save_pred_every == 0:
            print('###### Save model ######')
            torch.save(model.state_dict(), os.path.join(save_path, f'{i_iter + 1}.pth'))


if __name__ == '__main__':
    train()
