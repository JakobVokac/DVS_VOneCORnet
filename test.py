import pickle
import os
from PIL import Image
import numpy as np
from numpy.core.fromnumeric import transpose
from tqdm import tqdm
from dataset import CIFAR10
from torch.nn.functional import interpolate
from torchvision import transforms
from torchviz import make_dot

new_path = "d:/Datasets/cifar10resized/"

dataset = CIFAR10(new_path)

print(np.shape(dataset[0][0]))

sample = dataset[0][0]

totensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

sample = totensor(sample)
sample = normalize(sample)

exit()

path = "d:/Datasets/cifar10/cifar-10-batches-py/"
new_path = "d:/Datasets/cifar10resized/"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
files = os.listdir(path)

batch_files = []
for file in files:
    if file.startswith("data_batch"):
        file_path = os.path.join(path, file)
        batch_dict = unpickle(file_path)
        batch_files.append(batch_dict)
        
arr = batch_files[0][b'data'][0].reshape(3,32,32)
arr = np.transpose(arr, axes=[1, 2, 0])
arr = np.uint8(arr)

print(arr)
img = Image.fromarray(arr,"RGB")
print(img.size)
img.show()
img = img.resize((512,512),Image.BICUBIC)
img.show()
print(img.size)

arr2 = np.array(img)
arr2 = np.transpose(arr2, axes=[2, 0, 1])
print(arr2.shape)



path = "d:/Datasets/cifar10/cifar-10-batches-py/"
new_path = "d:/Datasets/cifar10resized/"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = os.listdir(path)

data = []
labels = []
for file in files:
    if file.startswith("data_batch"):
        file_path = os.path.join(path, file)
        batch_dict = unpickle(file_path)
        data.append(batch_dict[b'data'])
        labels.append(batch_dict[b'labels'])
    

data = np.concatenate(data)
labels = np.concatenate(labels)

print(np.shape(data))
print(np.shape(labels))

for i, img_path in tqdm(enumerate(data)):
    img = Image.fromarray(img_path)
    img = img.resize((512,512),Image.BICUBIC)
    file_idx = str(i)
    if len(file_idx) != 5:
        file_idx = "0"*(5-len(file_idx)) + file_idx
    np.savez(os.path.join(new_path,"x_train_"+file_idx+".npz"),img)

os.makedirs(new_path,exist_ok=True)
np.savez(os.path.join(new_path,"y_train.npz"),labels)

