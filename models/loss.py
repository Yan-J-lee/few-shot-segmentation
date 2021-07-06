import torch
import torch.nn as nn

# Loss parameters
dv = 0
dd = 2.5
gamma = 0.005


class ContrasiveLoss(nn.Module):
    '''
    A class designed purely to run a givan loss on a batch of samples.
    input is a batch of samples (as autograd Variables) and a batch of labels (ndarrays),
    output is the average loss (as autograd variable).
    '''
    def __init__(self):
        super(ContrasiveLoss, self).__init__()
        self.loss = contrasive_loss

    def forward(self, features_batch, labels_batch):
        """
        Args:
            features_batch: batch of feature embedding: [B, C, H, W]
            labels_batch: batch of labels: [B, H, W]
        """
        running_loss = 0.0
        batch_size = features_batch.shape[0]
        for i in range(batch_size):
            running_loss += self.loss(features_batch[i], labels_batch[i])

        ret = running_loss/(batch_size+1)
        return ret


def contrasive_loss(features, label):
    '''
    This loss is taken from "Semantic Instance Segmentation with a Discriminative Loss Function"
    by Bert De Brabandere, Davy Neven, Luc Van Gool at https://arxiv.org/abs/1708.02551

    :param features: a FloatTensor of (embedding_dim, h, w) dimensions.
    :param label: an nd-array of size (h, w) with ground truth instance segmentation. background is
                    assumed to be 0.
    :return: The loss calculated as described in the paper.
    '''
    label = label.flatten() # [HxW]
    features = features.permute(1,2,0).contiguous() # [H, W, C]
    H, W, C = features.size()
    features = features.view(H*W, C)

    instances = torch.unique(label)

    means = []
    var_loss = 0.0
    dist_loss = 0.0
    # calculate intra-cluster loss
    for instance in instances:
        if instance==255:   # ignore borders
            continue

        # collect all feature vector of a certain instance
        vectors = features[label == instance]
        size = vectors.shape[0]
        if size > 0:
            # get instance mean and distances to mean of all points in an instance
            mean = torch.mean(vectors, dim=0)
            dists = vectors - mean
            dist2mean = torch.sum(dists**2, dim=1)

            var_loss += torch.mean(dist2mean)
            means.append(mean)


    means = torch.stack(means)
    num_clusters = means.shape[0]
    for i in range(num_clusters):
        if num_clusters==1:  # no inter cluster loss
            break
        for j in range(i+1, num_clusters):
            dist = torch.norm(means[i]-means[j])
            if dist < dd*2:
                dist_loss += torch.pow(2*dd - dist,2) / (num_clusters-1)

    # regularization term
    reg_loss = torch.sum(torch.norm(means, 2, 1))
    total_loss = (var_loss + dist_loss + gamma*reg_loss) / num_clusters

    return total_loss