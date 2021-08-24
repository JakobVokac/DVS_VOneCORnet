import pickle
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

path = "d:/Datasets/cifar10/cifar-10-batches-py/"
new_path = "d:/Datasets/cifar10resized/"
os.makedirs(new_path,exist_ok=True)

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

for i, img_arr in tqdm(enumerate(data)):
    img_arr = img_arr.reshape(3,32,32)
    img_arr = np.transpose(img_arr, axes=[1, 2, 0])
    img = Image.fromarray(img_arr, "RGB")
    img = img.resize((128,128),Image.BICUBIC)
    resized_arr = np.array(img)
    resized_arr = np.transpose(resized_arr, axes=[2, 0, 1])
    file_idx = str(i)
    if len(file_idx) != 5:
        file_idx = "0"*(5-len(file_idx)) + file_idx
    np.savez(os.path.join(new_path,"x_train_"+file_idx+".npz"),resized_arr)

np.savez(os.path.join(new_path,"y_train.npz"),labels)

