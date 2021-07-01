"""
Few-Shot Image Semantic Segmentation Network Trained With Deep Metric Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vgg import Encoder
except:
    from vgg import Encoder

class MetricSegNet(nn.Module):
    """
    MetricSegNet: Few-Shot Image Semantic Segmentation Model Trained With Deep Metric Learning

    Args:
        in_channels: number of input channels
        pretrained_path: path of the pretrained model
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # feature extractor
        self.encoder = Encoder(in_channels, pretrained_path)

    def forward(self, support_imgs, qry_imgs):
        """
        Args:
            support_imgs: support images
                ways x shots x [B, 3, H, W], list of lists of Tensors
            qry_imgs: query images
                queries x [B, 3, H, W], list of Tensors 
        """
        n_ways, n_shots = len(support_imgs), len(support_imgs[0])
        n_queries = len(qry_imgs)
        B = support_imgs[0][0].shape[0]
        H, W = support_imgs[0][0].shape[-2:]

        ##### Feature extraction #####
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in support_imgs]
                                + [torch.cat(qry_imgs, dim=0)], dim=0) # [Bx(waysxshots + queries), 3, H, W]
        fts_concat = self.encoder(imgs_concat)
        Hf, Wf = fts_concat.shape[-2:]

        support_fts = fts_concat[:n_ways*n_shots*B] # [ways*shots*B, C, Hf, Wf]
        qry_fts = fts_concat[n_ways*n_shots*B:] # [queries*B, C, Hf, Wf]
        # upsample support_fts, qry_fts
        if self.training:
            support_fts = F.interpolate(support_fts, size=(H, W), mode='bilinear', align_corners=True) # [ways*shots*B, C, H, W]
            qry_fts = F.interpolate(qry_fts, size=(H, W), mode='bilinear', align_corners=True) # [ways*shots*B, C, H, W]
        return support_fts, qry_fts