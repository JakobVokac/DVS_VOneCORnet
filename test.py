
import os

from torch._C import device
from dataset import CIFAR10DVS
from dataset import VoxelGridDataset, CropTime
from tonic.transforms import Compose, ToVoxelGrid
from models import generate_vone_block, CORnet_Z_cifar10dvs
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block, out_channels = generate_vone_block(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
                                            simple_channels=256, complex_channels=256,
                                            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
                                            model_arch='resnet50', image_size=128, visual_degrees=8, ksize=25, stride=4)

block.to(device)
summary(block,(10,128,128))

model = CORnet_Z_cifar10dvs()

model.to(device)

summary(model,(10,128,128))