# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:12:58 2021

@author: korokoa

Inspired from the original pytorch implementation of imgenet training
"""

import os
import time
import torch
import socket
import argparse
import subprocess

import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


from torch.utils.tensorboard import SummaryWriter




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor,
             target: torch.Tensor,
             topk: Tuple[int] = (1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reduce_tensor(tensor: torch.Tensor, world_size: int):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def train(train_loader: DataLoader,
          model: nn.Module,
          criterion: nn.Module,
          optimizer: Optimizer,
          epoch: int,
          world_size: int):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits = model(input)
        loss = criterion(logits, target)

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

       # Measure accuracy
        prec1, prec5 = accuracy(logits.data, target.data, topk=(1, 5))

            # Average loss and accuracy across processes for logging
        reduced_loss = reduce_tensor(loss.data, world_size)
        prec1 = reduce_tensor(prec1, world_size)
        prec5 = reduce_tensor(prec5, world_size)

        # to_python_float incurs a host<->device sync
        batch_size = input[0].size(0)
        losses.update(to_python_float(reduced_loss), batch_size)
        top1.update(to_python_float(prec1), batch_size)
        top5.update(to_python_float(prec5), batch_size)

        torch.cuda.synchronize()
        batch_time.update((time.time() - end))
        end = time.time()

    return losses.avg,top1.avg,top5.avg,batch_time.sum


def adjust_learning_rate(initial_lr: float,
                         optimizer: Optimizer,
                         epoch: int):
    """Sets the learning rate to the initial LR decayed by sqrt 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** ((epoch) // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
	
    return lr



def validate(val_loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module,
             world_size: int):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                # compute output
                logits = model(input)
                loss = criterion(logits, target)

            # Measure accuracy
            prec1, prec5 = accuracy(logits.data, target.data, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data, world_size)
            prec1 = reduce_tensor(prec1, world_size)
            prec5 = reduce_tensor(prec5, world_size)

            # to_python_float incurs a host<->device sync
            batch_size = input[0].size(0)
            losses.update(to_python_float(reduced_loss), batch_size)
            top1.update(to_python_float(prec1), batch_size)
            top5.update(to_python_float(prec5), batch_size)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end))
            end = time.time()

         

    return losses.avg,top1.avg,top5.avg,batch_time.sum


def run(args):
     
    torch.manual_seed(args.seed)
    	
    batch_size=args.batch_size
    epochs=args.epochs
    learning_rate=args.lr
    save_model=args.save_model
    backend=args.backend
    
    
    # number of nodes / node ID
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())
    
    
     
    # set GPU device
    torch.cuda.set_device(local_rank)

    print("Initializing PyTorch distributed ...")
    torch.distributed.init_process_group(
        init_method='env://',
        backend=backend,
    )
    print(f"Backend: {dist.get_backend()}")
    
   
  
    print("Initialize Dataloaders...")
    
    transform = transforms.Compose(
        [transforms.Resize(size=(448,448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
    # Initialize Datasets.
    

    data = datasets.ImageFolder(root='/DATA/imagenet_images',
                                           transform=transform)

    classes=data.classes
    n_train = int(0.8*len(data))+1
    n_val = int(0.2*len(data))
    trainset, valset = torch.utils.data.random_split(data, [n_train, n_val])
    
    print(f"Number of training images: {len(trainset)}")
    print(f"Number of test images: {len(valset)}")
    print(f"Number of classes: {len(classes)}")
    
    #trainset = datasets.CIFAR100(root='./cifar100', train=True,
	#			download=False, transform=transform)
    
    #valset = datasets.CIFAR100(root='./cifar100', train=False,
		#		download=False, transform=transform)
	
    num_classes=len(classes)
    
    
    
    # Construct Model
    print("Initialize Model...")
    model = models.resnet18(pretrained=False, num_classes=num_classes).cuda()
    
    
    # Make model DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    
    
      

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
   
    

    # Create DistributedSampler to handle distributing the dataset across nodes
    # This can only be called after torch.distributed.init_process_group is called
    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)

    # Create the Dataloaders to feed data to the training and validation steps
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=10,
                              sampler=train_sampler,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            num_workers=10,
                            sampler=val_sampler,
                            pin_memory=True)
			    
			    
    if is_master:
        writer = SummaryWriter("imagenet_Nadam")
    
    
    best_prec1 = 0
    time=0
    best_epoch=0

    for epoch in range(epochs):
        # Set epoch count for DistributedSampler.
        # We don't need to set_epoch for the validation sampler as we don't want
        # to shuffle for validation.
        train_sampler.set_epoch(epoch)

        # Adjust learning rate according to schedule
        lr = adjust_learning_rate(learning_rate, optimizer, epoch)

        # train for one epoch
        train_loss,train_prec1,train_prec5,epoch_t1=train(train_loader, model, criterion, optimizer, epoch, world_size)

        # evaluate on validation set
        val_loss,val_prec1,val_prec5,epoch_t2 = validate(val_loader, model, criterion, world_size)
        time+=epoch_t1+epoch_t2

        # remember best prec@1 and save checkpoint if desired
        if val_prec1 > best_prec1:
            best_prec1 = val_prec1
            best_epoch=epoch
            if is_master and save_model:
                torch.save(model.state_dict(), "imagenet_Nadam_resnet18.pt")

        if is_master:
            print(f"Epoch {epoch+1} Summary: ")
            print(f"\tLearning rate: {lr}")
            print(f"\tTrain loss: {train_loss} ; Test loss: {val_loss}")
            print(f"\tTrain top1 accuracy: {train_prec1} ; Test top1 accuracy: {val_prec1}")
            print(f"\tTrain top5 accuracy: {train_prec5} ; Test top5 accuracy: {val_prec5}")
            print(f"\tTrain time: {epoch_t1} seconds; Test time:{epoch_t2} seconds")
            
            writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss,
            }, epoch+1)
            
            writer.add_scalars('Top1 accuracy', {
            'train': train_prec1,
            'val': val_prec1,
            }, epoch+1)
            
            
            writer.add_scalars('Top5 accuracy', {
            'train': train_prec5,
            'val': val_prec5,
            }, epoch+1)
	    
            writer.add_scalar("lr",lr,epoch)
    if is_master:
        print("\n")
        print("Training Summary:")
        print(f"\tBest epoch: {best_epoch}")
        print(f"\t Best Test top1 accuracy: {best_prec1}")
        print(f"\tTraining time: {time/60} minutes")
    



def main():
    parser = argparse.ArgumentParser(description='Imagenet Pytorch')
    
    parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
    
    parser.add_argument("--backend", type=str, default="nccl")
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=.1, metavar='LR',
                        help='learning rate (default: .1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    run(args)

	

	

if __name__ == "__main__":
	main()
  
   
