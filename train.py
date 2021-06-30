import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose
from pytorch_metric_learning import losses, distances
from torch.utils.tensorboard import SummaryWriter

from models.fewshot import FewShotSegNet
from models.metric import MetricSegNet
from dataset.voc import voc_fewshot
from dataset.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import config

def train():
    assert config.mode_type in ['fewshot', 'metric'] # sanity check
    # reproducibility
    set_seed(config.seed)

    # create model
    print('##### Create Model #####')
    if config.mode_type == 'fewshot':
        model = FewShotSegNet(pretrained_path=config.path['init_path']).cuda()
    elif config.mode_type == 'metric':
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
    dataset = voc_fewshot(
        base_dir=config.path['data_dir'],
        split=config.path['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=config.n_steps*config.batch_size,
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
        pin_memory=True
    )

    # set optimizer
    print('##### Set optimizer #####')
    optimizer = torch.optim.SGD(model.parameters(), **config.optim)
    scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)
    if config.mode_type == 'fewshot':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
    elif config.mode_type == 'metric':
        criterion = losses.TripletMarginLoss(distance=distances.CosineSimilarity())

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
                          for way in sample_batch['support_images']] # ways x shots x [B, 3, H, W]

        query_images = [query_image.cuda()
                        for query_image in sample_batch['query_images']] # queries x [B, 3, H, W]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batch['query_labels']], dim=0) # [queries*B, H, W]
        
        if config.mode_type == 'fewshot':
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batch['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                            for way in sample_batch['support_mask']]
            query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images) # [queries*B, ways+1, H, W]
            query_loss = criterion(query_pred, query_labels)
        elif config.mode_type == 'metric':
            support_labels = torch.cat([torch.cat(way, dim=0) for way in sample_batch['support_labels']]).long().cuda() # [Bxwaysxshots, H, W]
            support_fts, query_fts = model(support_images, query_images) # [ways*shots*B, C, Hf, Wf], [queries*B, C, Hf, Wf]
            # downsample support labels and query labels
            support_labels = F.interpolate(support_labels.unsqueeze(1).float(), size=support_fts.shape[-2:], mode='nearest').long().flatten() # [B*ways*shots*Hf*Wf]
            query_labels = F.interpolate(query_labels.unsqueeze(1).float(), size=query_fts.shape[-2:], mode='nearest').long().flatten() # [B*queries*Hf*Wf]
            # reshape support_fts and query_fts
            support_fts = support_fts.view(-1, support_fts.shape[1]).contiguous() # [B*ways*shots*Hf*Wf, C]
            query_fts = query_fts.view(-1, query_fts.shape[1]).contiguous() # [B*queries*Hf*Wf, C]
            # filter out unknown labels
            support_ind = (support_labels != config.ignore_label)
            support_labels, support_fts = support_labels[support_ind], support_fts[support_ind]

            query_ind = (query_labels != config.ignore_label)
            query_labels, query_fts = query_labels[query_ind], query_fts[query_ind]
            # random select n_sample samples in support_fts and query_fts
            support_rnd_ind = np.random.choice(len(support_labels), config.n_samples)
            support_labels, support_fts = support_labels[support_rnd_ind], support_fts[support_rnd_ind]

            query_rnd_ind = np.random.choice(len(query_labels), config.n_samples)
            query_labels, query_fts = query_labels[query_rnd_ind], query_fts[query_rnd_ind]
            query_loss = criterion(torch.cat((support_fts, query_fts), dim=0), torch.cat((support_labels, query_labels), dim=0))
        loss = query_loss
        # Forward and Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.item()
        logger.add_scalar('loss', query_loss, global_step=i_iter)
        log_loss['loss'] += query_loss

        # print loss and take snapshots
        if (i_iter + 1) % config.print_interval == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            print(f'step {i_iter+1}/{len(trainloader)}: loss: {loss}')

        if (i_iter + 1) % config.save_pred_every == 0:
            print('###### Save model ######')
            torch.save(model.state_dict(), os.path.join(save_path, f'{i_iter + 1}.pth'))

if __name__ == '__main__':
    train()