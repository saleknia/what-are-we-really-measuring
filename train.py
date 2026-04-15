import sys
# sys.path.append("/content/Scene-Recognition/models") 
# Instaling Libraries
import os
import copy
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
from torch.backends import cudnn

import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
import torch.optim as optim

from model.ConvNext import ConvNext
from model.VIT import VIT
from model.ResNet import ResNet
from model.MVIT import MVIT
from model.DinoV2 import DinoV2

from model.Mobile_netV2_loss import Mobile_netV2_loss

import utils
from utils import color
from utils import Save_Checkpoint
from trainer import trainer_func
from tester import tester_func
from config import *
from tabulate import tabulate
import warnings
from PIL import Image
from torchvision.transforms import InterpolationMode
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings('ignore')

from torchvision.transforms.functional import resize

import random
from PIL import Image

class CyclicResizeLoop:
    def __init__(self, resolutions, target_resolution=224):
        self.resolutions = resolutions
        self.target_resolution = target_resolution

    def __call__(self, img: Image.Image):
        # Pick a random start resolution from the loop
        start_res = random.choice(self.resolutions)
        
        # Find index of start and target in the loop
        start_idx = self.resolutions.index(start_res)
        target_idx = self.resolutions.index(self.target_resolution)
        
        # Build the cyclic path from start to target
        path = []
        idx = start_idx
        while True:
            path.append(self.resolutions[idx])
            if self.resolutions[idx] == self.target_resolution:
                break
            idx = (idx + 1) % len(self.resolutions)  # move forward cyclically
        
        # Resize step by step through path
        for res in path:
            img = img.resize((res, res), Image.BILINEAR)
        
        return img

res_loop = [112, 168, 224]
# res_loop = [168, 224, 280]

def main(args):

    # LOAD_DATA

    if TASK_NAME=='YCD':

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Resize((448, 448)),
            # transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomCrop(size=224, pad_if_needed=True),
            # CyclicResizeLoop(res_loop, target_resolution=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize((112, 112)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(root='/content/DC/train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/DC/valid' , transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1      , shuffle=False, num_workers=NUM_WORKERS)

        # trainset = torchvision.datasets.ImageFolder(root='/content/dataset/train', transform=transform_train)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

        # testset = torchvision.datasets.ImageFolder(root='/content/dataset/valid' , transform=transform_test)
        # test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1         , shuffle=False, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'test':test_loader}
    
    elif TASK_NAME=='YCDLW':

        transform_train = transforms.Compose([
            # # transforms.Resize((112, 112)),
            # transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomCrop(size=224, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=1.0),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize((112, 112)),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # trainset = torchvision.datasets.ImageFolder(root='/content/DataComp/content/DC/train', transform=transform_train)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # testset = torchvision.datasets.ImageFolder(root='/content/DataComp/content/DC/valid' , transform=transform_test)
        # test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1      , shuffle=False, num_workers=NUM_WORKERS)

        trainset = torchvision.datasets.ImageFolder(root='/content/dataset/train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/dataset/valid' , transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1      , shuffle=False, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'test':test_loader}

    elif TASK_NAME=='MIT-67':

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset     = torchvision.datasets.ImageFolder(root='/content/MIT-67-superclass/train/', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE , shuffle=True, num_workers=NUM_WORKERS)

        testset      = torchvision.datasets.ImageFolder(root='/content/MIT-67-superclass/test/' , transform=transform_test)
        test_loader  = torch.utils.data.DataLoader(testset , batch_size = 1          , shuffle=False, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'test':test_loader}

    elif TASK_NAME=='Stanford40':

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.ImageFolder(root='/content/StanfordActionDataset/train/',
                                        transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/StanfordActionDataset/test/', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size =  1, shuffle=True, num_workers=NUM_WORKERS)

        # NUM_CLASS = len(trainset.classes)

        data_loader={'train':train_loader,'test':test_loader}


    elif TASK_NAME=='OxfordIIITPet':

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.OxfordIIITPet(root='/content', split = 'trainval', transform = transform_train, download = True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.OxfordIIITPet(root='/content', split = 'test', transform = transform_test, download = True)
        test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1      , shuffle=False, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'test':test_loader}

    # MODEL_INITIALIZE

    if MODEL_NAME == 'ConvNext':
        model = ConvNext(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'MVIT':
        model = MVIT(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ResNet':
        model = ResNet(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'VIT':
        model = VIT(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'DinoV2':
        model = DinoV2(num_classes=NUM_CLASS).to(DEVICE)

    else: 
        raise TypeError('Please enter a valid name for the model type')

    # LOAD_MODEL

    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[[MODEL_NAME, f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )
    logger.info(model_table)

    if SAVE_MODEL:
        checkpoint = Save_Checkpoint(CKPT_NAME)
    else:
        checkpoint = None

    checkpoint_path = '/content/drive/MyDrive/checkpoint/'+CKPT_NAME+'_best.pth'  

    if LOAD_MODEL:
        logger.info('Loading Checkpoint...')
        if os.path.isfile(checkpoint_path):
            pretrained_model_path = checkpoint_path
            loaded_data = torch.load(pretrained_model_path, map_location=DEVICE)
            pretrained = loaded_data['net']
            model2_dict = model.state_dict()
            state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
            model2_dict.update(state_dict)
            model.load_state_dict(model2_dict)
        else:
            logger.info(f'No Such file : {checkpoint_path}')
        logger.info('\n')

    if args.train=='True':
        #######################################################################################################################################
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)  
        # optimizer      = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE) 
        total_batchs   = len(data_loader['train'])
        max_iterations = NUM_EPOCHS * total_batchs
        #######################################################################################################################################

        if POLY_LR is True:
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max_iterations, power=0.9)
        else:
            lr_scheduler =  None  

        # if POLY_LR is True:
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0)
        # else:
        #     lr_scheduler =  None 

    if args.train=='True':
        logger.info(50*'*')
        logger.info('Training Phase')
        logger.info(50*'*')
        loss_list = []
        for epoch in range(1, NUM_EPOCHS+1):
            trainer_func(
                epoch_num=epoch,
                model=model,
                dataloader=data_loader,
                optimizer=optimizer,
                device=DEVICE,
                ckpt=checkpoint,                
                num_class=NUM_CLASS,
                lr_scheduler=lr_scheduler,
                logger=logger,
                loss_list=loss_list)

        torch.save(loss_list , '/content/drive/MyDrive/loss_list.pt')

    if (args.inference=='True') and (os.path.isfile(checkpoint_path)):

        loaded_data = torch.load(checkpoint_path, map_location=DEVICE)
        pretrained  = loaded_data['net']
        model2_dict = model.state_dict()
        state_dict  = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}

        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)

        acc=loaded_data['acc']
        best_epoch=loaded_data['best_epoch']

        logger.info(50*'*')
        logger.info(f'Best Acc Over Validation Set: {acc:.2f}')
        logger.info(f'Best Epoch: {best_epoch}')

        logger.info(50*'*')
        logger.info('Inference Phase')
        logger.info(50*'*')
        tester_func(
            model=copy.deepcopy(model),
            dataloader=data_loader,
            device=DEVICE,
            ckpt=None,
            num_class=NUM_CLASS,
            logger=logger)

    logger.info(50*'*')
    logger.info('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--inference', type=str, default='True')
parser.add_argument('--train'    , type=str, default='True')
parser.add_argument('--KF'       , type=str, default='False')
parser.add_argument('--fold'     , type=str, default='0')

args = parser.parse_args()

def worker_init(worker_id):
    random.seed(SEED + worker_id)

if __name__ == "__main__":
    
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(SEED)    
    np.random.seed(SEED)  
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 

    if args.KF=='True':
        fold = int(args.fold)

    main(args)
    
    
