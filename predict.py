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



parser = argparse.ArgumentParser (description = "Set the parser for the training")

parser.add_argument('data_dir', help = 'Data directory (required)', type = str)
parser.add_argument('--lr', help = 'Learning rate', default=0.001, type = float)
parser.add_argument('--hidden_units', help = 'Number of classifier hidden units (as list [4096, 1024]', default=[4096, 1024], type = int)
parser.add_argument('--epochs', help = 'Number of epochs', default=5, type = int)
parser.add_argument('--GPU', help = "Option to use 'GPU' (yes/no)", default='yes', type = str)
parser.add_argument('--dropout', help = "Set dropout rate", default = 0.2)
parser.add_argument('checkpoint',help='Checkpoint of the model')
parser.add_argument('--top_k', type=int, help='Return top k most likely classes')
parser.add_argument('input', type=str, help='Image path')

args = parser.parse_args()

image_path = 'ImageClassifier/flowers/test/20/image_04912.jpg'

if args.input:
    image_path = args.input

checkpoint = load_checkpoint(filepath, cuda=False) 

if args.checkpoint:
    checkpoint = args.checkpoint
    

top_k = 1
if args.top_k:
    top_k = args.top_k
        
        
probabilities, classes = predict(image_path, model)
classes_names = [cat_to_name[number] for number in classes]
 
        
print("Probability confidence(s) = {}".format(probabilities))
print("Class(es) name(s) = {}".format(class_names))




def load_checkpoint(filepath, cuda=False):
    
    if not cuda:
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        checkpoint = torch.load(filepath)
        
    model = models.vgg16(pretrained = True)
    checkpoint = torch.load(filepath)

    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(4096, 102),
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model



def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    
    image = image_transform(Image.open(image))

    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cuda')
    
    img = process_image(image_path).to('cuda')
    np_img = img.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        logps = model.forward(np_img)
    
    ps = torch.exp(logps)
    top_k, top_classes_idx = ps.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])
    
    # Inverting dictionary
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    top_classes = []
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index])
    
    return list(top_k), list(top_classes)
