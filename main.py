from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import LeNet
from utils import progress_bar, save_model
import os

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--git-hash', type=bool, default=True, metavar='N',
                        help='whether or not to incldue git hash in model name')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def get_data(dataset, batch_size, kwargs):
    if dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader



def train(epoch, model, train_loader, optimizer, criterion, args):
    print(f"Epoch: {epoch}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, targets = Variable(data), Variable(target)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        batch_loss = train_loss/(batch_idx + 1)
        batch_acc  = 100. * correct/total
        progress_bar(batch_idx, len(train_loader),
                     f"Loss: {batch_loss:.3f} | Acc: {batch_acc:.3f}"
                     f"({correct}/{total})")

def test(epoch, model, test_loader, criterion, best_acc, args):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, targets = Variable(data, volatile=True), Variable(target)
        outputs = model(data)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        batch_loss = test_loss/(batch_idx + 1)
        batch_acc  = 100. * correct/total
        progress_bar(batch_idx, len(test_loader),
                     f"Loss: {batch_loss:.3f} | Acc: {batch_acc:.3f}"
                     f"({correct}/{total})")
    # Save checkpoint
    acc = 100. * correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'model': model.module if args.cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    test_loss = test_loss / len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return best_acc, test_loss

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader, test_loader = get_data('MNIST', args.batch_size, kwargs)
    model = LeNet()
    print(str(model).replace('\n', '').replace(' ', ''))
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    test_losses = []
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, criterion, args)
        best_acc, test_loss = test(epoch, model, test_loader, criterion, best_acc,
                                   args)
        test_losses.append(test_loss)
    save_model(model, best_acc, test_loss, epoch, args)

if __name__ == '__main__':
    main()
