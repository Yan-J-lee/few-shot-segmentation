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

from models.metric import MetricSegNet
from dataset.voc import voc_fewshot
from dataset.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import config

def train():
    # reproducibility
    set_seed(config.seed)

    # create model
    print('##### Create Model #####')
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
    criterion = losses.TripletMarginLoss(distance=distances.CosineSimilarity())

    # set logger
    logdir = os.path.join(config.path['log_dir'], f'{n_ways}_ways_{n_shots}_shots')
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
            [query_label.long().cuda() for query_label in sample_batch['query_labels']], dim=0).flatten() # [queries*B*H*W]
        
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batch['support_mask']]
        support_fg_mask = torch.cat([torch.cat(way, dim=0)
                            for way in support_fg_mask], dim=0).flatten()  # [B*ways*shots*H*W]
        support_fts, query_fts = model(support_images, query_images) # [ways*shots*B, C, H, W], [queries*B, C, H, W]
        # flatten support_fts and query_fts
        support_fts = support_fts.view(-1, support_fts.shape[1]).contiguous() # [B*ways*shots*H*W, C]
        query_fts = query_fts.view(-1, query_fts.shape[1]).contiguous() # [B*queries*H*W, C]
        # filter out unknown labels
        query_ind = (query_labels != config.ignore_label)
        query_labels, query_fts = query_labels[query_ind], query_fts[query_ind]
        # random select n_sample samples in support_fts for both positive and negative samples
        support_fg_pos, support_fg_neg = support_fg_mask[support_fg_mask == 1], support_fg_mask[support_fg_mask == 0]
        support_fts_pos, support_fts_neg = support_fts[support_fg_mask == 1], support_fts[support_fg_mask == 0]
        # positive samples for support images
        support_ind_pos = np.random.choice(len(support_fg_pos), config.n_samples)
        support_fg_pos, support_fts_pos = support_fg_pos[support_ind_pos], support_fts_pos[support_ind_pos]
        # negative samples for support images
        support_ind_neg = np.random.choice(len(support_fg_neg), config.n_samples)
        support_fg_neg, support_fts_neg = support_fg_neg[support_ind_neg], support_fts_neg[support_ind_neg]
        # random select n_sample samples in query_fts for both positive and negative samples
        query_labels_pos, query_labels_neg = query_labels[query_labels == 1], query_labels[query_labels == 0]
        query_fts_pos, query_fts_neg = query_fts[query_labels == 1], query_fts[query_labels == 0]
        # positive samples for query images
        query_ind_pos = np.random.choice(len(query_labels_pos), config.n_samples)
        query_labels_pos, query_fts_pos = query_labels_pos[query_ind_pos], query_fts_pos[query_ind_pos]
        # negative samples for query images
        query_ind_neg = np.random.choice(len(query_labels_neg), config.n_samples)
        query_labels_neg, query_fts_neg = query_labels_neg[query_ind_neg], query_fts_neg[query_ind_neg]
        
        query_loss = criterion(torch.cat((support_fts_pos, support_fts_neg, query_fts_pos, query_fts_neg), dim=0), 
                                torch.cat((support_fg_pos, support_fg_neg, query_labels_pos, query_labels_neg), dim=0))
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