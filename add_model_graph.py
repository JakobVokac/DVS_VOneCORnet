from dataset import VoxelGridDataset, CIFAR10, CIFAR10DVS, CropTime
from models import CORnet_Z_cifar10dvs, VOneCORnet_Z_cifar10dvs, CORnet_S, VOneCORnet_S
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from torchvision import transforms
from tonic.transforms import ToVoxelGrid, Compose
from torchviz import make_dot
from graphviz import Source

### MODEL & PATH ###

run_name = "cornets_cifar10dvs_snn_avgpool"
root_path = "d:/datasets/cifar10dvs/train/"
summary_path = "c:/Users/Jakob/Projects/v2e_vonenet/main/runs/"

current_summary_path = os.path.join(summary_path, run_name)
os.makedirs(current_summary_path,exist_ok=True)
writer = SummaryWriter(log_dir=current_summary_path,comment="CORnet-Z, CIFAR10DVS, 10 slice, average pooling")
model_path = os.path.join(current_summary_path, run_name + '.pth')
device = 'cpu'

model = CORnet_S(10).to(device)

lr = 1e-2
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adagrad(model.parameters(), lr=lr)

####################


### DATASET ###

dataset = VoxelGridDataset(root_path)
batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed= 42
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=train_sampler,num_workers=0,pin_memory=True)
validation_loader = DataLoader(dataset, batch_size=batch_size,
                                sampler=valid_sampler,num_workers=0,pin_memory=True)

batch_and_labels = next(iter(train_loader))
###############

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
s_epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Model loaded from epoch: ", s_epoch)

writer.add_graph(model, input_to_model=batch_and_labels[0], verbose=True)
writer.close()

y = model(batch_and_labels[0])

# dot = make_dot(y.mean(),params=dict(model.named_parameters()))
# dot.render("test.gv",view = True)
# source = Source(dot,filename="test.gv",format="png")
# source.view()