import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import time
import json


def arg_parser():
    parser = argparse.ArgumentParser
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help='Data directory')
    parser.add_argument('--image_path', type = str, help='Image to be predicted', default = 'flowers/test/1/image_06752.jpg')
    parser.add_argument('--topk', type = int, default=5, help='Top 5 probabilities displayed')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='Path to category mapping file')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint file', default = 'checkpoint.pth')
    parser.add_argument('--gpu', type = str, default = 'cuda', help = "Use GPU for training")
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print ("Loading checkpoint...")

def loading_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('output', nn.LogSoftmax(dim=1))]))
                    
    model = models.vgg19(pretrained=True)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model 

model = loading_checkpoint('checkpoint.pth')


print ("Processing image...")
def process_image(image):
    image = Image.open(image)
    
    transforming_image = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    image = np.array(transforming_image(image))

    return torch.from_numpy(image)


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


print ("Prediciting image...")

def predict(image_path, model, topk=5):
  
    model.to(device)
    model.class_to_idx = training_data.class_to_idx
    image_torch = process_image(image_path)
    image_torch = image_torch.to(device)
    model.eval()
    
    with torch.no_grad():
        image_torch = image_torch.unsqueeze_(0)
       
        output = model.forward(image_torch.float())
        ps = torch.exp(output)
        
        top_k, top_class = ps.topk(topk, dim=1)
        
        top_k = top_k.tolist()[0]

        model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        
        classes = [model.idx_to_class[idx] for idx in top_class[0].tolist()]
    
    return top_k, classes

print ("Sanity checking...")

def sanity_checking(image, model):
    
    total_classes = 5
    tensor_image = process_image(image)
    top_k, classes = predict(image, model)
    top_k = np.flip(top_k, axis = 0)
    classes = np.flip(classes, axis = 0)
    labels = [cat_to_name[c1] for c1 in classes]
    np_image = tensor_image.numpy().squeeze()
    imshow(process_image(image))
    
    fig, (X, Y) = plt.subplots(figsize=(20,20), nrows=2)
    X.axis('off')    
    
    Y.barh(np.arange(len(top_k)), top_k)
    Y.set_yticks(np.arange(total_classes))
    Y.set_yticklabels(labels)
    Y.set_title("Top 5 Probabilities")
    Y.set_xlim(0,1)
    
sanity_checking("flowers/test/1/image_06752.jpg",model)
