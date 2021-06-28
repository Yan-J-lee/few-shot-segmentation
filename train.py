import enum
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

from models.fewshot import FewShotSegNet
from dataset.voc import voc_fewshot
from dataset.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import config

def train():
    # reproducibility
    set_seed(config.seed)

    # create model
    print('##### Create Model #####')
    model = FewShotSegNet(pretrained_path=config.path['init_path']).cuda()
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
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

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
    for i_iter, sampled_batch in enumerate(trainloader):
        # Prepare input 
        support_images = [[shot.cuda() for shot in way]
                          for way in sampled_batch['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sampled_batch['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sampled_batch['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sampled_batch['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sampled_batch['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss
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