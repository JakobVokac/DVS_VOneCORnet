import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np

class CIFAR10DVSDataset(Dataset):
    def __init__(self, img_dir):
        self.img_labels = np.concatenate([[(i,x) for i in range(1000)] for x in range(10)])
        self.img_classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_classes[self.img_labels[idx][1]] + "/cifar10_" + self.img_classes[self.img_labels[idx][1]] + "_" + str(self.img_labels[idx][0])
        img_paths = [img_path + "_frame_" + str(fr) + ".png" for fr in range(10)]
        image = torch.cat([read_image(img_paths[i]) for i in range(10)],dim=0)
        label = self.img_labels[idx][1]
        return image, label
        
