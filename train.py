import time
from argparse import Namespace
from matplotlib import pyplot as plt
# used for logging to TensorBoard
from tensorboard_logger import log_value

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import densenet as dn
from utils import *

classes = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def train(train_loader: torch.utils.data.DataLoader, model: dn.DenseNet3,
          criterion: nn.CrossEntropyLoss, optimizer: torch.optim.SGD, epoch: int, args: Namespace):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.perf_counter()
    for i, (inp, target) in enumerate(train_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        output: torch.Tensor = model(inp)
        loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, inp.size(0))
        top1.update(prec1, inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
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
    for i, (inp, target) in enumerate(val_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        with torch.no_grad():
            output: torch.Tensor = model(inp)
            loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        prec1: torch.Tensor = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, inp.size(0))
        top1.update(prec1, inp.size(0))

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def test(test_loader: torch.utils.data.DataLoader, model: dn.DenseNet3,
         args: Namespace):
    """Perform testing on the test set"""
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
                if i == args.batch_size:
                    break
            else:
                count += 1
                inv_trans = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[255. / x for x in [63.0, 62.1, 66.7]]),
                    transforms.Normalize(mean=[- x / 255. for x in [125.3, 123.0, 113.9]], std=[1., 1., 1.])
                ])
                images = inv_trans(images)
                print(f"Example [{count}]:")
                print(f"Prediction: {classes[predicted[i].item()]}")
                print(f"Label: {classes[labels[i].item()]}\n")
                plt.imshow(images[i].permute(1, 2, 0).cpu())
                plt.show()

    print(f'Accuracy of the network on the 10000 test images: {top1.avg:.2f}%')
