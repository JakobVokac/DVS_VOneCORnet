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

load_model = False
run_name = "cornets_cifar10dvs_snn_avgpool_swapped_layers"
root_path = "d:/datasets/cifar10dvs/train/"
summary_path = "c:/Users/Jakob/Projects/v2e_vonenet/main/runs/"


def run():
    current_summary_path = os.path.join(summary_path, run_name)
    os.makedirs(current_summary_path,exist_ok=True)
    writer = SummaryWriter(log_dir=current_summary_path,comment="CORnet-Z, CIFAR10DVS, 10 slice, average pooling, swapped layers")
    model_path = os.path.join(current_summary_path, run_name + '.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # dataset = CIFAR10DVSDataset(root_path)
    # dataset = CIFAR10DVS( "d:/Datasets/cifar10dvs/",transform=Compose(
    #     [CropTime(0,1e+6), ToVoxelGrid(10)]
    # ))
    
    dataset = VoxelGridDataset(root_path)
    # transform = transforms.Compose(
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = CIFAR10(root_path,transform=transform)
    
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
                              sampler=train_sampler,num_workers=4,pin_memory=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                   sampler=valid_sampler,num_workers=4,pin_memory=True)


    model = CORnet_S(10).to(device)

    lr = 1e-2
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)

    n_epochs = 100
    s_epoch = 0
    if load_model:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        s_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Model loaded from epoch: ", s_epoch)
        
    old_accuracy = 0
    val_acc_counter = 0
    err_margin = 0.001
    for epoch in np.arange(s_epoch+1, n_epochs):
        print("Epoch: ", epoch)
        
        model.train()
        train_accuracy = torch.tensor(0).to(device)
        train_loss = torch.tensor(0,dtype=torch.float32).to(device)
        for batch, labels in tqdm(train_loader):

            batch, labels = batch.to(device, dtype=torch.float32), labels.to(device, dtype=torch.int64)
            
            pred = model(batch)
            
            opt.zero_grad()
            out = loss(pred,labels)
            train_loss = train_loss + out
            out.backward()
            opt.step()
            batch_acc = torch.sum(torch.eq(torch.argmax(pred,dim=1), labels))
            train_accuracy = train_accuracy + batch_acc
            
        train_accuracy = train_accuracy.cpu().numpy()
        train_accuracy_percentage = train_accuracy/(dataset_size - split)
        print("Training Accuracy: ", train_accuracy, len(train_loader)*batch_size, dataset_size, train_accuracy_percentage)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy_percentage, epoch)
        model.eval()
            
        accuracy = torch.tensor(0).to(device)
        # total_loss = torch.tensor(0).to(device)
        for batch, labels in validation_loader:
            
            batch, labels = batch.to(device, dtype=torch.float32), labels.to(device, dtype=torch.int64)

            pred = model(batch)
            # loss_ce = loss(pred,labels)
            batch_acc = torch.sum(torch.eq(torch.argmax(pred,dim=1), labels))
            accuracy = accuracy + batch_acc
            # total_loss = total_loss + loss_ce
            
        accuracy = accuracy.cpu().numpy()/split
        print("Validation Accuracy: ", len(validation_loader)*batch_size, split, accuracy)
        writer.add_scalar("Accuracy/validation", accuracy, epoch)
        if accuracy <= old_accuracy + err_margin:
            val_acc_counter += 1
        else:
            val_acc_counter = 0    
        old_accuracy = accuracy
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss,
                    }, 
                model_path)
        writer.flush()
        print("Model saved at epoch: ", epoch)
        if val_acc_counter == 3:
            break;


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()
    