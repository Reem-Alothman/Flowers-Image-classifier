
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json
import seaborn as sb
import argparse


def get_dataloaders(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                   


    train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    
    return trainloader, validloader, testloader


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def train_model(trainloader,validloader,learning_rate = 0.001,cuda = "cuda",epochs = 5):
    epochs = epochs
    print_every = 64
    steps = 0
    
    model = models.vgg16(pretrained=True)
    
    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(4096, 102),
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(cuda)

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
        
            steps += 1
            inputs, labels = inputs.to(cuda), labels.to(cuda)
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                print("Epoch: {}/{}   ".format(e+1, epochs),
                      "Training Loss: {:.4f}   ".format(running_loss/print_every),
                      "Valid Loss: {:.4f}   ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.4f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                model.train()
            
    return model

def validation(model, validloader, criterion):
    model.to('cuda')
    valid_loss = 0
    accuracy = 0
    for data in validloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def save_checkpoint(model):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'arch':'vgg16',
                  'class_to_idx':model.class_to_idx,
                  'model_state_dict': model.state_dict()}

    torch.save(checkpoint,'checkpoint.pth')
    
    
    
    
    
    



parser = argparse.ArgumentParser(description='Train a new network on a data set')

parser.add_argument('data_dir', type=str, help='Path of the Image Dataset (with train, valid and test folders)')
parser.add_argument('--learning_rate', type=float, help='Learning rate. Default is 0.01')
parser.add_argument('--epochs', type=int, help='Number of epochs.')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
args, _ = parser.parse_known_args()

data_dir = 'flowers'
if args.data_dir:
    data_dir = args.data_dir


learning_rate = 0.01
if args.learning_rate:
    learning_rate = args.learning_rate


epochs = 5
if args.epochs:
    epochs = args.epochs
    
    
cuda = False
if args.gpu:
    if torch.cuda.is_available():
        cuda = True
    else:
        print("Warning! GPU flag was set however no GPU is available in the machine")


trainloader, validloader, testloader = get_dataloaders(data_dir)

  
if test_loaders:
    images, labels = next(iter(trainloader))
    imshow(images[2])
    plt.show()

    images, labels = next(iter(validloader))
    imshow(images[2])
    plt.show()

    images, labels = next(iter(testloader))
    imshow(images[2])
    plt.show()

model = train_model(trainloader, validloader,learning_rate, cuda, epochs)

save_checkpoint(model)


    
