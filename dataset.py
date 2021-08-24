import os
import numpy as np
from tonic.datasets.dataset import Dataset as TonicDataset
from aedat_loader import loadaerdat
import itertools
from torch.utils.data import Dataset

class CropTime:
    """Crops events with timestamps outside a specified time. This class is made to work as a Tonic Transform.
    The time measure might be specific to the dataset you're using. For the CIFAR10DVS dataset, time is in us.
    This function does not work (do anything) with images. The time interval is [start,stop].
    
    Parameters:
        start (int): Starting timepoint. Crops everything before this timepoint.
        stop (int): End timepoint. Crops everything after this timepoint.
    """
    
    
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = events[events[:,0] >= self.start]
        events = events[events[:,0] <= self.stop]
        return events, images



class CIFAR10DVS(TonicDataset):

    sensor_size = (128, 128)
    ordering = "txyp"

    def __init__(
        self, base_path, save_to = "./", download=False, transform=None, target_transform=None
    ):
        super(CIFAR10DVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.base_path = base_path
        self.labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        files = [(os.listdir(self.base_path + label), label) for label in self.labels]
        files = [[self.base_path + label + "/" + file for file in file_list] for (file_list, label) in files]
        items = [[(file, label) for file in files[label] if file.endswith(".aedat")] for label in range(len(files))]
        self.items = list(itertools.chain.from_iterable(items))
        self.len = len(self.items)
        
    def __getitem__(self, index):
        filepath, target = self.items[index]
        t,x,y,p = loadaerdat(filepath)
        events = np.stack((t,x,y,p)).T

        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
            
        return events, target, filepath

    def __len__(self):
        return self.len

class VoxelGridDataset(TonicDataset):
    
    def __init__(
        self, base_path, save_to = "./", download=False, transform=None, target_transform=None
    ):
        super(VoxelGridDataset, self).__init__(
            save_to, transform=None, target_transform=None
        )
        self.base_path = base_path
        self.labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        files = [(os.listdir(self.base_path + label), label) for label in self.labels]
        files = [[self.base_path + label + "/" + file for file in file_list] for (file_list, label) in files]
        items = [[(file, label) for file in files[label] if file.endswith(".npz")] for label in range(len(files))]
        self.items = list(itertools.chain.from_iterable(items))
        self.len = len(self.items)
        
    def __getitem__(self, index):
        filepath, target = self.items[index]
        voxel_grid = np.load(filepath)
        return voxel_grid["arr_0"], target

    def __len__(self):
        return self.len

class CIFAR10(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)
        self.files.remove("y_train.npz")
        self.labels = np.load(os.path.join(root_dir,"y_train.npz"))["arr_0"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        sample = np.load(os.path.join(self.root_dir, self.files[idx]))["arr_0"]
        sample = np.transpose(sample, axes=[1,2,0])
        target = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, target