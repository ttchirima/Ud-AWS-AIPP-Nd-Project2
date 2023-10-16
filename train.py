import argparse
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session

class Train_class:
    @staticmethod
    def initialize(data_dir):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
       
        train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        test_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        
        image_datasets = {}
        image_datasets["train"] = datasets.ImageFolder(train_dir, transform = train_data_transforms)
        image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
        image_datasets["test"] = datasets.ImageFolder(test_dir, transform = test_data_transforms)

        train_dataloader = torch.utils.data.DataLoader(image_datasets["train"], batch_size = 64, shuffle =True)
        valid_dataloader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size = 32)
        test_dataloader = torch.utils.data.DataLoader(image_datasets["test"], batch_size = 32)
        
        print(f"Data Source: {data_dir} directory.")
        return image_datasets, train_dataloader, valid_dataloader, test_dataloader
    
    @staticmethod
    def create_model(arch,h_u):
        if arch.lower() == "vgg16":
            model = models.vgg16(pretrained=True)
        else:
            model = models.densenet121(pretrained=True)
        
        for p in model.parameters():
            p.requires_grad = False 
        
        if arch.lower() =='vgg16':
            classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(25088,h_u)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(h_u, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(1024,h_u)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(h_u, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        model.classifier = classifier
        print(f"Model built using {arch} and {hidden_units} hidden units.")
        return model        
    
    def check_validation(model, dataloader, criterion, device): 
        loss = 0
        accuracy = 0
        with torch.no_grad(): 
            for images, labels in iter(dataloader):
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss += criterion(output,labels).item()
                ps = torch.exp(output)
                equality = (labels.data == ps.max(dim = 1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()
        return loss,accuracy 
    
    @staticmethod
    def train_model(model, train_dataloader, valid_dataloader, learning_rate, epochs, gpu):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
        
        device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        model.to(device)
        print_ev = epochs
        running_loss = 0
        steps = 0
        running_loss = 0
        train_accuracy = 0

        with active_session():
            for x in range(epochs):
                model.train()
                for images, labels in iter(train_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    steps += 1
                    optimizer.zero_grad()
                    output = model.forward(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim = 1)[1])
                    train_accuracy += equality.type(torch.FloatTensor).mean()
        
                    if steps % print_ev == 0:
                        model.eval()
                
                        with torch.no_grad():
                            valid_lo , valid_acc = Train_class.check_validation(model, valid_dataloader, criterion,device)
                    
                            print("E: {}/{}.. ".format(x+1, epochs),
                                "T_Loss: {:.3f}.. ".format(running_loss/print_ev),
                                "T_Accuracy: {:.3f}".format(train_accuracy/print_ev),
                                "V_Loss: {:.3f}.. ".format(valid_lo/len(valid_dataloader)),
                                "V_Accuracy: {:.3f}".format(valid_acc/len(valid_dataloader)))
                            running_loss = 0
                            train_accuracy = 0
                            model.train()
    
            print("Done with the Training!")
        return model, optimizer, criterion
    
     
'''usage: python train.py data_directory Prints out training loss, validation loss, and validation accuracy as the network trains
 Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg16"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
'''   
    
# Get the command line input into the scripts
parser = argparse.ArgumentParser()

# Command to run code: python train.py data_directory
parser.add_argument('data_directory', action='store',
                    default = 'flowers',
                    help='Set directory to load training data')

# Command to set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints')

# Command to choose architecture: python train.py data_dir --arch "vgg16"
parser.add_argument('--arch', action='store',
                    default = 'vgg16',
                    dest='arch',
                    help='Choose architecture: e.g., "densenet121"')

# Command to set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 10
parser.add_argument('--learning_rate', action='store',
                    default = 0.01,
                    dest='learning_rate',
                    help='Choose architecture learning rate')

parser.add_argument('--hidden_units', action='store',
                    default = 512,
                    dest='hidden_units',
                    help='Choose architecture hidden units')

parser.add_argument('--epochs', action='store',
                    default = 10,
                    dest='epochs',
                    help='Choose architecture number of epochs')

# Command to use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')

parse_results = parser.parse_args()
data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

# Load and preprocess data
train_obj = Train_class()
image_datasets, train_loader, valid_loader, test_loader = train_obj.initialize(data_dir)

# Building and training the classifier
model_init = train_obj.create_model(arch, hidden_units)
model, optimizer, criterion = train_obj.train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"

print(f'Checkpoint saved to {save_dir_name}.')