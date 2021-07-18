from models import CORnet_Z_cifar10dvs
import numpy as np
import torch
from data import CIFAR10DVSDataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

load_model = True
model_path = "c:/Users/Jakob/Projects/v2e_vonenet/main/model_save_state/model.pth"
root_path = "d:/datasets/cifar10dvs/"

def run():
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    dataset = CIFAR10DVSDataset(root_path)

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


    model = CORnet_Z_cifar10dvs().to(device)

    lr = 1e-2
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)

    n_epochs = 1000
    s_epoch = 0
    #Total data is roughly 16GB, my gpu has 8GB memory, so im splitting up the loading to 4 chunks
    if load_model:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        s_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Model loaded from epoch: ", s_epoch)
        
    for epoch in np.arange(s_epoch+1, n_epochs):
        print("Epoch: ", epoch)
        
        model.train()
        train_accuracy = torch.tensor(0).to(device)
        for batch, labels in train_loader:
            batch, labels = batch.to(device, dtype=torch.float32), labels.to(device, dtype=torch.int64)
            
            pred = model(batch)
            
            opt.zero_grad()
            out = loss(pred,labels)
            out.backward()
            opt.step()
            batch_acc = torch.sum(torch.eq(torch.argmax(pred,dim=1), labels))
            train_accuracy = train_accuracy + batch_acc
            
        train_accuracy = train_accuracy.cpu().numpy()
        print("Training Accuracy: ", train_accuracy, len(train_loader)*batch_size, dataset_size, train_accuracy/(dataset_size - split))
        if (epoch % 10 == 0):
            accuracy = torch.tensor(0).to(device)
            # total_loss = torch.tensor(0).to(device)
            for batch, labels in validation_loader:
                
                batch, labels = batch.to(device, dtype=torch.float32), labels.to(device, dtype=torch.int64)

                pred = model(batch)
                # loss_ce = loss(pred,labels)
                batch_acc = torch.sum(torch.eq(torch.argmax(pred,dim=1), labels))
                accuracy = accuracy + batch_acc
                # total_loss = total_loss + loss_ce
                
            accuracy = accuracy.cpu().numpy()
            print("Validation Accuracy: ", len(validation_loader)*batch_size, split, train_accuracy/split)
            
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                        }, 
                    model_path)
            print("Model saved at epoch: ", epoch)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()
    