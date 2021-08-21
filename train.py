import matplotlib.pyplot as plt
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


print("Command line arguments...")
def parser():
    parser = argparse.ArgumentParser
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help='Data directory')
    parser.add_argument('--arch', type = str, default = 'vgg19', help = "Data architecture")
    parser.add_argument('--learningrate', type = float, default = 0.01, help = "Gradient descent")
    parser.add_argument('--hidden_units', type = int, default = 512, help = "Number of hidden units")
    parser.add_argument('--epochs', type = int, default = 20, help = "Number of epochs")
    parser.add_argument('--image_path', type = str, help='Image to be predicted', default = 'flowers/test/1/image_06752.jpg')
    parser.add_argument('--topk', type = int, default=5, help='Top 5 probabilities displayed')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='Path to category mapping file')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint file', default = 'checkpoint.pth')
    parser.add_argument('--gpu', type = str, default = 'cuda', help = "Use GPU for training")
    args = parser.parse_args()
    return args


print ("Transforms and dataloaders..")

data_dir = args.dir
train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'
test_dir = data_dir + 'test'

training_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


validating_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

testing_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 



training_data = datasets.ImageFolder(train_dir, transform = training_transform)

validating_data = datasets.ImageFolder(valid_dir, transform = validating_transform)

testing_data = datasets.ImageFolder(test_dir, transform = testing_transform )

                                                            
trainloader = torch.utils.data.DataLoader(training_data, batch_size= 64, shuffle = True)

validloader = torch.utils.data.DataLoader(validating_data, batch_size= 64, shuffle=True)

testloader = torch.utils.data.DataLoader(testing_data, batch_size = 64, shuffle=True)
                                                            

                                                           
print("Label mapping..")   
                                                            
import json
 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print(cat_to_name)                                                           
                                                            
  
print ("Training the classifier...")                                                                                                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.vgg19(pretrained = True)
model

                                                            
for param in model.parameters():
    param.requires_grad = False
    

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('output', nn.LogSoftmax(dim=1))]))             
for param in model.parameters():
    param.requires_grad = False
 
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
classifier                                                        

def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy
                                                            
                                                            
epochs = args.epochs
steps = 0
print_every = 5
model.to(device)

for epoch in range(epochs):
    running_loss = 0
    model.train()
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
    
        output = model.forward(inputs)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
            
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Accuracy: {:.3f}".format(accuracy/len(validloader)))

accuracy = 0
model.eval()

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                                                            


print ("Saving checkpoint..")
model.class_to_idx = training_data.class_to_idx

checkpoint = {'classifier': model.classifier,
              'epochs': epochs, 
              'device': device,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer_dict': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')                                                            
                                                            
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
                                               
