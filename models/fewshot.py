"""
Few-Shot Image Semantic Segmentation Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vgg import Encoder
except:
    from vgg import Encoder

class FewShotSegNet(nn.Module):
    """
    FewShotSegNet: Few-Shot Image Semantic Segmentation Model

    Args:
        in_channels: number of input channels
        pretrained_path: path of the pretrained model
        cfg: model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # feature extractor
        self.encoder = Encoder(in_channels, pretrained_path)

    def forward(self, support_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            support_imgs: support images
                ways x shots x [B, 3, H, W], list of lists of Tensors
            fore_mask: foreground masks for support images
                ways x shots x [B, H, W], list of lists of Tensors
            back_mask: background masks for support images
                ways x shots x [B, H, W], list of lists of Tensors
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

        support_fts = fts_concat[:n_ways*n_shots*B].view(n_ways, n_shots, B, -1, Hf, Wf) # [ways, shots, B, C, Hf, Wf]
        qry_fts = fts_concat[n_ways*n_shots*B:].view(n_queries, B, -1, Hf, Wf) # [queries, B, C, Hf, Wf]
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0) # [ways, shots, B, H, W]
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0) # [ways, shots, B, H, W]

        ##### Make pixel-wise prediction for query images based on distance in feature space #####
        outputs = []
        for episode in range(B):
            # extract prototypes for each way
            support_fg_fts = [
                    [self.get_features(support_fts[way, shot, [episode]], fore_mask[way, shot, [episode]]) 
                    for shot in range(n_shots)] 
                for way in range(n_ways)] # ways x shots x [1, C]
            support_bg_fts = [
                    [self.get_features(support_fts[way, shot, [episode]], back_mask[way, shot, [episode]])
                    for shot in range(n_shots)]
                for way in range(n_ways) # ways x shots x [1, C]
            ]

            # obtain foreground and background prototypes by averaging the prototypes among shots
            fg_prototypes, bg_prototypes = self.get_prototype(support_fg_fts, support_bg_fts) # (ways)x[1, C]  [1, C]

            # compute distance
            prototypes = [bg_prototypes,] + fg_prototypes # (ways+1)x[1, C]
            dist = [self.calculate_dist(qry_fts[:, episode], prototype) for prototype in prototypes] # (ways+1)x[queries, Hf, Wf]
            pred = torch.stack(dist, dim=1) # [queries, way+1, Hf, Wf]
            outputs.append(F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True))

        output = torch.stack(outputs, dim=1) # [queries, B, ways+1, H, W]
        output = output.view(-1, *output.shape[2:])
        return output

    def get_features(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask.unsqueeze(1), dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8) # [1, C]
        return masked_fts

    def get_prototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: ways x shots x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: ways x shots x [1 x C]

        Returns:
            fg_prototypes: list of forground prototypes for each way
                shape: ways x [1 x C]
            bg_prototype: background prototypes
                shape: [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def calculate_dist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: [queries, C, Hf, Wf]
            prototype: prototype of one semantic class
                expect shape: [1, C]
        """
        dist = F.cosine_similarity(fts, prototype.unsqueeze(-1).unsqueeze(-1), dim=1) * scaler
        return dist

# sanity check
if __name__ == '__main__':
    net = FewShotSegNet().cuda()
    support_imgs = [[torch.rand(2, 3, 128, 128).cuda()], [torch.rand(2, 3, 128, 128).cuda()]]
    fore_mask = [[torch.rand(2, 128, 128).cuda()], [torch.rand(2, 128, 128).cuda()]]
    back_mask = [[torch.rand(2, 128, 128).cuda()], [torch.rand(2, 128, 128).cuda()]]
    qry_imgs = [torch.rand(2, 3, 128, 128).cuda()]
    out = net(support_imgs, fore_mask, back_mask, qry_imgs)
    print(out.shape)