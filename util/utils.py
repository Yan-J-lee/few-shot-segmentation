"""Util functions"""
import random
import torch

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    }
}