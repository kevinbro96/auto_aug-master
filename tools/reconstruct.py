from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys

sys.path.append('.')

from aa.networks import *
import aa.config as cf
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC


def reconst_images(epoch=2, batch_size=128, batch_num=2, dataloader=None, model=None):

    cifar10_dataloader = dataloader

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                bs = X.size(0)
                 _, xi, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X.data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(xi.data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X - xi).data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X-Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, testloader):
    # set model as testing mode
    model.eval()
    # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
    acc_avg = AverageMeter()
    sparse_avg = AverageMeter()
    share_mag = AverageMeter()
    top1 = AverageMeter()
    top1_x_xi = AverageMeter()
    top1_xi = AverageMeter()
    TC = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(100, -1)), p=2, dim=1)
            hi, xi, mu, logvar = model(x)
            acc_xi = 1 - F.mse_loss(torch.div(xi, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_xd = 1 - F.mse_loss(torch.div(x - xi, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100

            acc_avg.update(acc_xi.data.item(), bs)

            sparse_avg.update(acc_xd.data.item(), bs)

            tc = total_correlation(hi, mu, logvar) / bs / args.dim
            TC.update(tc.item(), bs)

        wandb.log({'acc_avg': acc_avg.avg, \
                   'sparse_avg': sparse_avg.avg, \
                   'test-TC': TC.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tAcc: %.4f TC: %.4f" % (epoch, acc_avg.avg, TC.avg))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=model)


def train(args, epoch, model, optimizer, trainloader):
    model.train()
    model.training = True

    loss_avg = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=args.alpha)
        x, y, y_b = x.cuda(), y.cuda().view(-1, ), y_b.cuda().view(-1, )
        x, y = Variable(x), [Variable(y), Variable(y_b)]
        bs = x.size(0)
        optimizer.zero_grad()

        _, xi, mu, logvar = model(x)

        l1 = F.mse_loss(xi, x)
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * args.dim
        loss = args.re * l1  + args.kl * l3
        loss.backward()
        optimizer.step()


        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)


        n_iter = (epoch - 1) * len(trainloader) + batch_idx
        wandb.log({'loss': loss_avg.avg, \
                   'loss_rec': loss_rec.avg, \
                   'loss_kl': loss_kl.avg}, step=n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_kl: %.4f %%'
                % (epoch, args.epochs, batch_idx + 1,
                   len(trainloader), loss_avg.avg, loss_rec.avg,  loss_kl.avg,))

    if epoch % 10 == 1:
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_new.pth'))  # save motion_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(args.save_dir, 'optimizer_new.pth'))  # save optimizer

        print("Epoch {} model saved!".format(epoch + 1))


def main(args):
    learning_rate = 1.e-3
    learning_rate_min = 2.e-4
    CNN_embed_dim = args.dim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch, batch_size, optim_type = cf.start_epoch, cf.batch_size, cf.optim_type
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('\n[Phase 2] : Model setup')
    if args.model == "nonorm":
        model = RCVAE_nonorm(d=8, z=CNN_embed_dim)
    elif args.model == "og":
        model = RCVAE_s1(d=CNN_embed_dim)

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * np.ceil(50000 / batch_size),
                                                           eta_min=learning_rate_min)


    if args.testOnly:
        model.load_state_dict(torch.load("results/emb32_5.0_0.5/model_new.pth"))
        model.eval()
        test_loss = 0
        # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
        loss_avg = AverageMeter()
        share_mag = AverageMeter()
        top1 = AverageMeter()
        top1_x_xi = AverageMeter()
        top1_xi = AverageMeter()
        TC = AverageMeter()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                # distribute data to device
                x, y = x.cuda(), y.cuda().view(-1, )
                bs = x.size(0)
                hi, xi, mu, logvar = model(x)
                loss = 1 - F.mse_loss(torch.div(xi, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                      torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                      reduction='sum') / 100
                test_loss += loss.item()  # sum up batch loss
                loss_avg.update(loss.data.item(), bs)
                # measure accuracy and record loss

                tc = total_correlation(hi, mu, logvar) / bs / CNN_embed_dim
                TC.update(tc.item(), bs)
            print("\n| Validation \t\tLoss: %.4f TC: %.4f" % (loss_avg.avg, TC.avg))
            print("| X: %.2f%% X-Xi: %.2f%% Xi: %.2f%%" % (top1.avg, top1_x_xi.avg, top1_xi.avg))
        sys.exit(0)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        train(args, epoch, model, optimizer, trainloader)
        scheduler.step()
        if epoch % 10 == 1:
            test(epoch, model, testloader)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    wandb.finish()
    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--model', default='nonorm', type=str, help='model name')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--dim', default=8, type=int, help='CNN_embed_dim')
    parser.add_argument('--re', default=0.5, type=float, help='reconstruction weight')
    parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
    parser.add_argument('--ce', default=1.0, type=float, help='cross entropy weight')
    args = parser.parse_args()
    wandb.init(config=args)
    set_random_seed(args.seed)
    main(args)