# few-shot-segmentation
few shot segmentation

### Dependencies

* Python 3.6 +
* PyTorch 1.7.0
* torchvision 0.8.0
* pytorch-metric_learning 0.9.99
* NumPy, SciPy, PIL
* tensorboard

### Data Preparation for VOC Dataset

1. Download `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and put them under `VOCdevkit/VOC2012`.

2. Download `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and use it to replace `VOCdevkit/VOC2012/ImageSets/Segmentation`.


### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `PANet/pretrained_model` folder.

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. 