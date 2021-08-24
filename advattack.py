import os
import torch
from models import CORnet_S
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from dataset import VoxelGridDataset, CIFAR10
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms



run_name = "cornets_cifar10_resized"
root_path = "d:/datasets/cifar10resized/"
summary_path = "c:/Users/Jakob/Projects/v2e_vonenet/main/runs/"
current_summary_path = os.path.join(summary_path, run_name)
model_path = os.path.join(current_summary_path, run_name + '.pth')

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CIFAR10(root_path,transform=transform)

batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
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

sample, target = next(iter(validation_loader))
x_test = sample.numpy()
y_test = target.numpy()

    
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_test)
print(y_test)
print(type(x_test))
print(type(y_test))
print(x_test.shape)
print(y_test.shape)
model = CORnet_S(3)

lr = 1e-2
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adagrad(model.parameters(), lr=lr)

checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
s_epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Model loaded from epoch: ", s_epoch)


classifier = PyTorchClassifier(
    model=model,
    loss=loss,
    optimizer=opt,
    input_shape=(3, 128, 128),
    nb_classes=10,
)


predictions = classifier.predict(x_test)
print(predictions.shape)
print(y_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))