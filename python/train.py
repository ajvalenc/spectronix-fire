import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *

feature_extract = False
num_classes = 2


def train_val_dataset(dataset, val_split=0.20):
    """Split a dataset into a train and validation set."""
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def train_resnet101_on_our_data():
    """Train a ResNet-101 on our data."""
    data_dir = "/home/ajvalenc/Datasets/spectronix/thermal/fire/processed"
    #dataset = ImageFolder(data_dir, transform=Compose([Resize((224, 224)), ToTensor()]))
    dataset = ImageFolder(data_dir, transform=Compose([ToTensor()])) #ImageFolder internally converts PIL images to rgb
    datasets = train_val_dataset(dataset)
    train_dataloader = DataLoader(datasets['train'], batch_size=4, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(datasets['val'], batch_size=4,shuffle=True, num_workers=1)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = models.resnet101(pretrained=True)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    def accuracy(out, labels):
        """Get accuracy of predictions for a batch"""
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()

    def set_parameter_requires_grad(model, feature_extracting):
        """Set requires_grad=False for all layers except the last fc layer"""
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    set_parameter_requires_grad(net, feature_extract)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    net.fc = net.fc.to(device)

    n_epochs = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)

    # Train loop
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)

            if (batch_idx) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in test_dataloader:
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(test_dataloader))
            #network_learned = batch_loss < valid_loss_min
            network_learned = epoch > 9
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                # torch.save(net.state_dict(), 'resnet.pt')
                torch.save(net, 'resnet101_10epoch.pth')
                print('Improvement-Detected, save-model')
        net.train()

    plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')

print("File one __name__ is set to: {}" .format(__name__))

if __name__ == '__main__':
    train_resnet101_on_our_data()

