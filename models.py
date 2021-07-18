import numpy as np
from vonenet.vonenet import VOneBlock
from vonenet.params import generate_gabor_param
from collections import OrderedDict
from torch import nn

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

# def generate_vone_block(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
#             simple_channels=256, complex_channels=256,
#             noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
#             model_arch='resnet50', image_size=128, visual_degrees=8, ksize=25, stride=4):
    
#     out_channels = simple_channels + complex_channels

#     sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

#     # gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
#     #                 'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
#     #                 'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
#     # arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}
    
#     ppd = image_size / visual_degrees

#     sf = sf / ppd
#     sigx = nx / sf
#     sigy = ny / sf
#     theta = theta/180 * np.pi
#     phase = phase / 180 * np.pi

#     vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
#                            k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
#                            simple_channels=simple_channels, complex_channels=complex_channels,
#                            ksize=ksize, stride=stride, input_size=image_size)

#     return vone_block, out_channels

def CORnet_Z_cifar10dvs():
    
    #vone_block, out_channels = generate_vone_block(simple_channels=32, complex_channels=32, ksize=7, stride=2)
    
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(10, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 10)),
            ('output', Identity())
        ])))
    ]))
    
    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

