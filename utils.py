from argparse import Namespace
import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import densenet as dn

# used for logging to TensorBoard
from tensorboard_logger import log_value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader: torch.utils.data.DataLoader, model: dn.DenseNet3, 
    criterion: nn.CrossEntropyLoss, optimizer: torch.optim.SGD, epoch: int, args: Namespace):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.perf_counter()
    for i, (input, target) in enumerate(train_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        input: torch.Tensor = input.cuda()

        # compute output
        output: torch.Tensor = model(input)
        loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader: torch.utils.data.DataLoader, model: dn.DenseNet3, 
    criterion: nn.CrossEntropyLoss, epoch: int, args: Namespace):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    end = time.perf_counter()
    for i, (input, target) in enumerate(val_loader):
        target: torch.Tensor = target.cuda(non_blocking = True)
        input: torch.Tensor = input.cuda()

        # compute output
        with torch.no_grad():
            output: torch.Tensor = model(input)
            loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        prec1: torch.Tensor = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

def test(test_loader: torch.utils.data.DataLoader, model: dn.DenseNet3, args: Namespace):
    """Perform testing on the test set"""
    classes = ["airplane", "automobile", "bird", "cat", "deer", 
                "dog", "frog", "horse", "ship", "truck"]
    count = 0
    top1 = AverageMeter()

    model.eval()

    for (images, labels) in test_loader:
        labels: torch.Tensor = labels.cuda(non_blocking=True)
        images: torch.Tensor = images.cuda()
        
        with torch.no_grad():
            outputs: torch.Tensor = model(images)
        
        _, predicted = torch.max(outputs.cuda().data, 1)
        prec1: torch.Tensor = accuracy(outputs.data, labels, topk=(1,))[0]
        top1.update(prec1, images.size(0))

        if args.test and count < 10:
            i = 0
            while torch.equal(predicted[i], labels[i]):
                i += 1
                if i == args.batch_size: break
            else:
                count += 1
                invTrans = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[255. / x for x in [63.0, 62.1, 66.7]]),
                    transforms.Normalize(mean=[- x / 255. for x in [125.3, 123.0, 113.9]], std=[1., 1., 1.])
                    ])
                images = invTrans(images)
                print(f"Example [{count}]:")
                print(f"Prediction: {classes[predicted[i].item()]}")
                print(f"Label: {classes[labels[i].item()]}\n")
                plt.imshow(images[i].permute(1, 2, 0).cpu())
                plt.show()

    print(f'Accuracy of the network on the 10000 test images: {top1.avg:.2f}%')

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res