import utils
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import print_progress
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
from valid import valid_func
from torcheval.metrics import MulticlassAccuracy
from torch.nn.modules.loss import CrossEntropyLoss
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)  # 2N x D
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # 2N x 2N

        # Create labels
        N = self.batch_size
        labels = torch.arange(N, device=z_i.device)
        labels = torch.cat([labels, labels], dim=0)

        # Mask to exclude self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positives = torch.cat(
            [torch.diag(similarity_matrix, N), torch.diag(similarity_matrix, -N)], dim=0
        )

        nominator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)

        loss = -torch.log(nominator / denominator).mean()
        return loss

warnings.filterwarnings("ignore")

def trainer_func(epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,logger,loss_list):
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to(DEVICE)
    model.train()

    loss_ce_total   = utils.AverageMeter()
    loss_co_total   = utils.AverageMeter()

    # accuracy = utils.AverageMeter()
    metric = MulticlassAccuracy(average="macro", num_classes=num_class).to(DEVICE)
    # accuracy = mAPMeter()

    loss_ce = CrossEntropyLoss(label_smoothing=0.0)
    loss_co = nn.MSELoss() # NTXentLoss(batch_size=64, temperature=0.5)

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    base_iter      = (epoch_num-1) * total_batchs
    iter_num       = base_iter

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()
        
        outputs = model(inputs, targets.long())

        ce_loss = loss_ce(outputs, targets.long()) 
        co_loss = 0.0 # 1000.0 * loss_co(F.normalize(features[0], p=2, dim=1), F.normalize(features[1], p=2, dim=1)) 

        loss = ce_loss + co_loss

        loss_list.append(loss)
        loss_ce_total.update(ce_loss)
        loss_co_total.update(co_loss)

        predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
        metric.update(predictions, targets.long())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            # suffix=f'CE_Loss = {loss_ce_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',   
            suffix=f'CE_loss = {loss_ce_total.avg:.4f} , CO_loss = {loss_co_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',                 
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.avg:.4f}',   
            bar_length=45
        )  
  
    Acc = 100 * metric.compute()
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_ce_total.avg:.4f}, Accuracy = {Acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')

    if ckpt is not None:
       ckpt.save_best(acc=Acc, epoch=epoch_num, net=model)

        