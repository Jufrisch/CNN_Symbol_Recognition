'''
Code adapted from: 
https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

As part of UF's EEL5840 Final project for handwritten image recognition

Justin Frisch
'''

import sys
from PIL import Image
import numpy as np
import torch
from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.utils.data.dataset import random_split
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss

    def validation_step(self,batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class SymbolClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(1, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(512,25)
        )
    
    def forward(self, xb):
        return self.network(xb)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, 'b')
    plt.plot(val_losses, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy vs. No. of epochs')
    plt.show()


def main():
    with open('../project/dataset_new/Train_Images.npy','rb') as f:
        img = np.load(f)

    with open('../project/dataset_new/Train_Labels.npy','rb') as f:
        lab = np.load(f).astype(int)

    img = np.reshape(img,[img.shape[0],1,150,150])
    lab = np.reshape(lab,[img.shape[0]])

    print(f"{img.shape = }")
    print(f"{lab.shape = }")

    tensor_x = torch.Tensor(img)
    tensor_y = torch.LongTensor(lab)
    tensor_x = tensor_x.to(device='cuda')
    tensor_y = tensor_y.to(device='cuda')

    dataset = TensorDataset(tensor_x,tensor_y) 

    batch_size = 32
    val_size = 2350 # 90/10 split
    train_size = lab.shape[0] - val_size
    train_data,val_data = random_split(dataset,[train_size,val_size])

    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = False)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 0, pin_memory = False)

    model = SymbolClassification()
    model.to(device='cuda')
    num_epochs = 100
    opt_func = torch.optim.Adam
    lr = 0.0001
    #fitting the model on training data and record the result after each epoch
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


    plot_accuracies(history)
    plot_losses(history)

    torch.save(model.state_dict(),"test_model")

if __name__ == '__main__':
    main()
