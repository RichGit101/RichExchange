## 16 Jul Command line prg for Flower classification AIP project 1,1
###############################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse
import json
from collections import OrderedDict

###############################################
def get_input_args():
    print("\n iTrace Welcome to Flower classifier Project \n")
    parser = argparse.ArgumentParser(description="\nWelcome to Flower classifier Project\n")
    parser.add_argument('--data_dir', type=str,default = 'flowers',help='Directory to Flowers Dataset,Please <flowers> ')
    parser.add_argument('--gpu_req', type=str,default ='No',help='Please choose if GPU to be used <No>:')
    parser.add_argument('--epochs', type=int,default =2, help='Please choose no of epochs <2> :')
    parser.add_argument('--arch', type=str,default='vgg19', help='Please choose <vgg>, alexnet,resnet : ')
    parser.add_argument('--learn_rate', type=float,default=0.9, help='Please enter learning rate <0.9> :')
    parser.add_argument('--uts_hid', type=int,default=5, help='Please enter hidden units <5>:')
    parser.add_argument('--check_pt', type=str,default='mdlckpt', help='File name for Trained Model Check Point persistance for Transfer,please <mdlckpt>:')
   ##Check and redo 
    arg_inp = parser.parse_args() 
    return arg_inp
###############################################
def check_command_line_arguments(in_arg):
    # prints command line agrs
    #
    print("Command Line Arguments Read:\n",     
          "\n Directory for models =", in_arg.data_dir, 
          "\n GPU (depends on applicability) =", in_arg.gpu_req, 
          "\n No of Epochs requested =", in_arg.epochs,
          "\n Learning rate requested =", in_arg.learn_rate,
          "\n Hidden units requested =",in_arg.uts_hid,
          "\n Model archtecture selected =", in_arg.arch, 
          "\n File name for Trained Model Check Point persistance for Transfer =", in_arg.check_pt)
    # Check GPU and default
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################
def mtransLoad(dataset_dir, phase):
    print("\n iTrace Begins - inside mtransLoad in ttrainer \n")
    print("\n iTrace Begins - Normalising images with mean and SD for converting the values of each color channel to be between -1 and 1 instead of 0 and 1. \n")
    print("\niTrace Rubric - one random scaling, Rotation, mirroring or cropping to (224x224 pix) in transform dictionary for three keys depending on training or validation\n")
    ##Check Phase and set training /testing/validation transform
     ##Check Phase and set training /testing/validation data sub dir
     
## if phase=='train':
    print("\niTrace Training Phase and entered inside of train phase in mtransLoad \n")
    train_transforms = transforms.Compose([
    transforms.RandomRotation(43),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ## trg_dir = dataset_dir + '/train/'
    ## print("\niTrace Training Phase in mtransLoad trg dir is  %s\n",trg_dir)
    
    ## train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms) 
    ## trainloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
    ## loaded_datasets = trainloader
## if phase=='valida':
    print("\niTrace Validation Phase and entered inside of train phase in mtransLoad \n")
    valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    ## valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    ## valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=2)
    ## loaded_datasets = valid_loader
   ## if phase=='test':
    print("\niTrace Testing Phase and entered inside of train phase in mtransLoad \n")
    test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    ## test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    ## test_loader = torch.utils.data.DataLoader(test_data, batch_size=2)
    ## loaded_datasets = test_loader
## elif:
    print("\niTrace Invalid Phase and now in else of phase in mtransLoad \n")
    ## create image dataset dictionary img_dsd
    img_dsd = {'train':'','test':'','valid':''}
    trg_val =datasets.ImageFolder(root=dataset_dir + '/train',transform=train_transforms)
    tst_val =datasets.ImageFolder(root=dataset_dir + '/test',transform=test_transforms)
    vali_val =datasets.ImageFolder(root=dataset_dir + '/valid',transform=valid_transforms)
    img_dsd['train']= trg_val
    img_dsd['test']= tst_val
    img_dsd['valid']= vali_val
    print("\niTrace verify image data set dictionary dump \n")
    print(img_dsd) 
    
    ########################################################
    print("\niTrace Rubric - Loading data and Batch sizing \n")
    
    trainloader = data.DataLoader(img_dsd['train'], batch_size=4, shuffle=True, num_workers=2)
    testloader = data.DataLoader(img_dsd['test'], batch_size=4, num_workers=2)
    validloader = data.DataLoader(img_dsd['valid'], batch_size=4, num_workers=2)
    
    
    print("\niTrace completed - Loading data and Batch sizing completed\n")

    ## trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    ## train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    ## test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    ## trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    ## testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    ## loaded_datasets = 'x'
        
    return img_dsd

###############################################
    
def load_ptmdl(arch_req):
               
  ##      arch='vgg19', num_labels=102, hidden_units=4096):
    # Load a pre-trained model
    ##model = models.densenet121(pretrained=True)
    print("\niTrace Loading Pre trained model- requested arch :\n",arch_req)
    if arch_req =='vgg19':
        print("\niTrace Loading Pre trained model in progress :\n",arch_req)
        ptmodel = models.vgg19(pretrained=True)
        print("\niTrace Loading Pre trained model completed :\n",arch_req)
    elif arch_req =='alexnet':
        print("\niTrace Loading Pre trained model in progress :\n",arch_req)
        ptmodel = models.alexnet(pretrained=True)
        print("\niTrace Loading Pre trained model completed :\n",arch_req)
    else:
        raise ValueError('Unexpected network architecture, current options are <vgg19> or alexnet', arch_req)
      ## check at start at cmd line options
    print("\niTrace returning Pre trained model- requested arch :",arch_req) 
    return ptmodel
###############################################
######################
   #######run model on archi and note time
   
   

### stitch classofier based on last layer of selected arch
   ## return model.classifier

def adj_classi (orglmodel):
    print("\niTrace Stitching params for classifier in adj_classi :\n")
    print("\niTrace Freezing params,No back propagation \n")
    ## Load and note note their classifier features in every pre trained
    ## based on it, prepare classifier with in and out features for our case
    ## assign to classifier based on model
    ## for dense net its a single layer linear classifier with 1024 in and 1000 out
    ## for vgg16 its a 6 layer 25088 in linear at 0th classifier layer
    for param in orglmodel.parameters():
        param.requires_grad = False
    ## 10 dimenstions for test train hence last layer is 10    
    nclassifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 15000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(15000, 8000)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(8000, 800)),
                          ('relu', nn.ReLU()),
                          ('fc4', nn.Linear(800, 10)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    orglmodel.classifier = nclassifier
    print("\niTrace stiching classifier completed \n")
###############################################
def train_model():
    print("\niTrace In model taining fun train_model  :\n")
    device = 'cpu'
    criterion = nn.NLLLoss()
    




###############################################
## def train_model(lr,epoch, gpu):




###############################################
def load_ccatg (jsonf):
    print("\n iTrace Begins - begining JSON loader load_ccatg \n")
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
###############################################

# Define command line arguments
# Creates parse 
print("\n iTrace Begins - begining before main inside TTrainer \n")
## def main():
print("\n iTrace Begins - Start of main inside TTrainer \n")
## start_time = time()
in_arg = get_input_args()
check_command_line_arguments(in_arg)
    
    
    
# Train model if invoked from command line
 
if in_arg.data_dir: 
    print("\n iTrace yes inside data cmd line dir check\n")
    phase = 'train'
    mtransLoad(in_arg.data_dir, phase) 
else:
    print("\n iTrace yes inside data cmd line dir ELSE check\n")
print("\n iTrace Begins - Call Json label mapper inside TTrainer \n")
cat_to_name = 'cat_to_name.json'
load_ccatg(cat_to_name)
arch_req=in_arg.arch
pre_trgloaded =load_ptmdl(arch_req)
adj_classi(pre_trgloaded)  
print("\n iTrace End of prg\n")       