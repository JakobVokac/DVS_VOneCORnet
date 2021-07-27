
import os
from dataset import CIFAR10DVS
from dataset import VoxelGridDataset, CropTime
from tonic.transforms import Compose, ToVoxelGrid



dataset = VoxelGridDataset( "d:/Datasets/cifar10dvs/train/")
train_data = CIFAR10DVS(("d:/Datasets/cifar10dvs/train/"),transform=Compose(
        [CropTime(0,1e+6), ToVoxelGrid(10)]
    ))

sample, target = dataset[0]
sample2, target2, filepath2 = train_data[0]
print(sample)
print(sample2)