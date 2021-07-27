
from dataset import CIFAR10DVS, CropTime
from tonic.transforms import ToVoxelGrid, Compose, FlipLR, FlipUD, FlipPolarity
import os
import numpy as np
from tqdm import tqdm

###
# This code assumes you have downloaded and unpacked the CIFAR10DVS dataset into a 'root_path' that you will be using for training/testing the model and haven't changed it yet in anyway.
# It splits the splits each class in the dataset to train/test = 9/1 ('split' variable) and separates the folders into train/test subfolders in the root directory.
# Then it takes both the train dataset and the test dataset and crops out the events after 1e+6 ('stop_time') microseconds (since the events are too sparse after 1e+6 us).
# It then transforms each aedat file in both datasets into a voxelgrid with 10 'slices', saved as a numpy array (.npz). The model in main then works with these files only.
#
# Then optionally, it takes the training dataset (.aedat) and applies transformations to create new training samples (horizontal flip, vertical flip, polarity flip).
#
# NOTE: this process can take up to an hour or more.
###

stop_time = 1e+6
slices = 10
split = .9
root_path = "d:/Datasets/cifar10dvs/"
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# # Make new folders and move the files
# os.makedirs(root_path + "train")
# [os.makedirs(root_path + "train/" + img_class) for img_class in classes]
# os.makedirs(root_path + "test")
# [os.makedirs(root_path + "test/" + img_class) for img_class in classes]

# files = [(img_class, os.listdir(root_path + img_class)) for img_class in classes]
# for img_class, folder in files:
#     folder_len = len(folder)

#     for file in folder[:int(split*folder_len)]:
#         os.rename(os.path.join(root_path, img_class, file), os.path.join(root_path, "train", img_class, file))
        
#     for file in folder[int(split*folder_len):]:
#         os.rename(os.path.join(root_path, img_class, file), os.path.join(root_path, "test", img_class, file))
        
# # Turn events to voxel grids
# train_data = CIFAR10DVS(os.path.join(root_path, "train/"),transform=Compose(
#         [CropTime(0,stop_time), ToVoxelGrid(slices)]
#     ))

# test_data = CIFAR10DVS(os.path.join(root_path, "test/"),transform=Compose(
#         [CropTime(0,stop_time), ToVoxelGrid(slices)]
#     ))

# for sample in tqdm(train_data):
#     events, target, filepath = sample
#     new_filepath = filepath[:-6] + '.npz'
#     np.savez(new_filepath, events)
    
# for sample in tqdm(test_data):
#     events, target, filepath = sample
#     new_filepath = filepath[:-6] + '.npz'
#     np.savez(new_filepath, events)
    

# Make new training samples and turn them to voxel grids
train_data = CIFAR10DVS(os.path.join(root_path, "train/"),transform=Compose(
        [CropTime(0,stop_time), FlipLR(), ToVoxelGrid(slices)]
    ))

for sample in tqdm(train_data):
    events, target, filepath = sample
    new_filepath = filepath[:-6] + '_lr.npz'
    np.savez(new_filepath, events)
    
train_data = CIFAR10DVS(os.path.join(root_path, "train/"),transform=Compose(
        [CropTime(0,stop_time), FlipUD(), ToVoxelGrid(slices)]
    ))

for sample in tqdm(train_data):
    events, target, filepath = sample
    new_filepath = filepath[:-6] + '_ud.npz'
    np.savez(new_filepath, events)
    
train_data = CIFAR10DVS(os.path.join(root_path, "train/"),transform=Compose(
        [CropTime(0,stop_time), FlipPolarity(1.0), ToVoxelGrid(slices)]
    ))

for sample in tqdm(train_data):
    events, target, filepath = sample
    new_filepath = filepath[:-6] + '_pol.npz'
    np.savez(new_filepath, events)

