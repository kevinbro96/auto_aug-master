from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import datetime
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('.')

from aa.networks import *
import aa.config as cf
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--save_dir', default='./results/autoaug_embedding/', type=str, help='save_dir')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--seed', default=666, type=int, help='seed')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
alpha = 2.0
CNN_embed_dim = 32  # latent dim extracted by 2D CNN
epochs = 300
learning_rate = 1.e-3
learning_rate_min = 2.e-4
kl_coef = 1.0
ce_coef = 1.0
ls_coef = 1.5
alpha = 2.0
args = parser.parse_args()
save_path=args.save_dir
set_random_seed(args.seed)
setup_logger(args.save_dir)
writer = SummaryWriter(args.save_dir)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, batch_size, optim_type = cf.start_epoch, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugmentMC(n=2, m=10),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
net.apply(conv_init)
model = CVAE_s1_e(d=CNN_embed_dim, num_channels=3)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)
optimizer_c = AdamW(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * np.ceil(50000 / batch_size),
                                                       eta_min=learning_rate_min)
scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=epochs * np.ceil(50000 / batch_size),
                                                         eta_min=learning_rate_min)
criterion = nn.CrossEntropyLoss()

def reconst_images(epoch=2, batch_size=128, batch_num=3, train=True, model=None, save_path='./imgs'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                                    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset])])

    # cifar10 dataset (images and labels)
    if train:
        if args.dataset =='cifar10':
            cifar10_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False,
                                                           transform=transform)
        else:
            cifar10_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=False,
                                                            transform=transform)
        datasource = 'train'
    else:
        if args.dataset =='cifar10':
            cifar10_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False,
                                                           transform=transform)
        else:
            cifar10_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False,
                                                            transform=transform)
        datasource = 'test'

    cifar10_dataloader = torch.utils.data.DataLoader(dataset=cifar10_dataset, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # if model is None:
    #     # model = torch.nn.DataParallel(
    #     #     models['cifar10']['vae'](d=CNN_embed_dim, k=num_classes, kl_coef=kl_coef, ce_coef=ce_coef,
    #     #                              num_channels=3).to(device), device_ids=[0, 1])
    #     model = models['cifar10']['vae'](d=CNN_embed_dim, k=num_classes, kl_coef=kl_coef, ce_coef=ce_coef,
    #                                      num_channels=3).to(device)
    #     model.load_state_dict(torch.load(os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch))))
    #     print('VQVAE epoch {} model reloaded!'.format(epoch))
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                bs = X.size(0)
                _, xi, xd, xh,_, _ = model(X)

                grid_X = torchvision.utils.make_grid(X.data,  nrow=8, padding=2, normalize=True)
                writer.add_image( '_Batch_{batch}_{datasource}_X.jpg'.format(batch=batch_idx,datasource=datasource),grid_X,epoch)
                grid_Xi = torchvision.utils.make_grid(xi.data,  nrow=8, padding=2, normalize=True)
                writer.add_image('Batch_{batch}_{datasource}_Xi.jpg'.format( batch=batch_idx,datasource=datasource),grid_Xi,epoch)
                grid_Xd = torchvision.utils.make_grid(xd.data,  nrow=8, padding=2, normalize=True)
                writer.add_image('_Batch_{batch}_{datasource}_Xd.jpg'.format(batch=batch_idx,datasource=datasource),grid_Xd,epoch)
                grid_Xh = torchvision.utils.make_grid((xh).data,  nrow=8, padding=2, normalize=True)
                writer.add_image( '_Batch_{batch}_{datasource}_Xh.jpg'.format( batch=batch_idx,datasource=datasource),grid_Xh,epoch)
                grid_X_Xh = torchvision.utils.make_grid((X-xh).data,  nrow=8, padding=2, normalize=True)
                writer.add_image( '_Batch_{batch}_{datasource}_X-Xh.jpg'.format( batch=batch_idx,datasource=datasource),grid_X_Xh,epoch)
    print('reconstruction complete!')

# Training
def train(epoch):
    net.train()
    net.training = True
    model.train()
    model.training = True

    loss_avg = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    loss_sparse = AverageMeter()
    top1 = AverageMeter()


    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=alpha)
        x, y, y_b = x.cuda(), y.cuda().view(-1, ), y_b.cuda().view(-1, )
        x, y = Variable(x), [Variable(y),Variable(y_b)]
        bs = x.size(0)

        optimizer.zero_grad()

        _, xi, xd, xh, mu, logvar = model(x)
        l1 = F.mse_loss(xh, x)
        entropy = (F.softmax(net(xi), dim=1) * F.log_softmax(net(xi), dim=1)).sum(dim=1).mean()
        cross_entropy = lam * F.cross_entropy(net(xd), y[0]) + (1. - lam) * F.cross_entropy(net(xd), y[1])
        l2 = cross_entropy + entropy
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * 1024
        l4 = (xd * xd).mean()
        loss = l1 + ce_coef * l2 + kl_coef * l3 + ls_coef * l4
        loss.backward()
        optimizer.step()
        #update classifer
        optimizer_c.zero_grad()
        out = net(xd.detach())
        l5 = lam * F.cross_entropy(out, y[0]) + (1. - lam) * F.cross_entropy(out, y[1])
        l5.backward()
        optimizer_c.step()

        # measure accuracy and record loss
        prec1, prec5, correct, pred = accuracy(out.data, y[0].data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_entropy.update(entropy.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)
        loss_sparse.update(l4.data.item(), bs)
        top1.update(prec1.item(), bs)

        n_iter = epoch * len(trainloader) + batch_idx
        writer.add_scalar('loss', loss_avg.avg, n_iter)
        writer.add_scalar('loss_rec', loss_rec.avg, n_iter)
        writer.add_scalar('loss_ce', loss_ce.avg, n_iter)
        writer.add_scalar('loss_entropy', loss_entropy.avg, n_iter)
        writer.add_scalar('loss_kl', loss_kl.avg, n_iter)
        writer.add_scalar('loss_sparse', loss_sparse.avg, n_iter)
        writer.add_scalar('acc', top1.avg, n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Loss_sparse: %.4fAcc@1: %.3f%%'
                    %(epoch, epochs, batch_idx+1,
                        len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg, loss_kl.avg, loss_sparse.avg, top1.avg))
            #sys.stdout.flush()
    if epoch % 10 == 9:
        torch.save(model.state_dict(),
                   os.path.join(save_path, 'model_new.pth'))  # save motion_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(save_path, 'optimizer_new.pth'))  # save optimizer
        torch.save(net.state_dict(),
                   os.path.join(save_path, 'net_new.pth'))  # save motion_encoder
        torch.save(optimizer_c.state_dict(),
                   os.path.join(save_path, 'optimizer_c_new.pth'))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, train=True, model=model, save_path=save_path)
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, train=False, model=model, save_path=save_path)


def test(epoch):
    # set model as testing mode
    model.eval()
    net.eval()
    test_loss = 0
    # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
    loss_avg = AverageMeter()
    share_mag = AverageMeter()
    top1 = AverageMeter()
    top1_x_xi = AverageMeter()
    top1_xi = AverageMeter()
    top1_xd = AverageMeter()
    TC = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)

            hi, xi, xd, xh, mu, logvar = model(x)
            out = net(x)
            out_i = net(xi)
            out_h = net(xh)
            out_d = net(xd)
            loss = F.mse_loss(xi + xd, x)
            test_loss += loss.item()  # sum up batch loss
            share_mag.update(torch.abs(xd).norm() / torch.abs(x).norm(), bs)
            loss_avg.update(loss.data.item(), bs)
            # measure accuracy and record loss
            prec1, _,_,_ = accuracy(out.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

            prec1_x_xi, _,_,_ = accuracy(out_h.data, y.data, topk=(1, 5))
            top1_x_xi.update(prec1_x_xi.item(), bs)

            prec1_xi,  _,_,_ = accuracy(out_i.data, y.data, topk=(1, 5))
            top1_xi.update(prec1_xi.item(), bs)

            prec1_xd,  _,_,_ = accuracy(out_d.data, y.data, topk=(1, 5))
            top1_xd.update(prec1_xd.item(), bs)

            tc = total_correlation(hi, mu, logvar)/  bs / 32
            TC.update(tc.item(), bs)

        writer.add_scalar('/test/loss', loss_avg.avg, epoch)
        writer.add_scalar('/test/Share-Mag', share_mag.avg, epoch)
        writer.add_scalar('/test/X-acc', top1.avg, epoch)
        writer.add_scalar('/test/X-Xi-acc', top1_x_xi.avg,epoch)
        writer.add_scalar('/test/Xi-acc', top1_xi.avg,epoch)
        writer.add_scalar('/test/Xd-acc', top1_xd.avg,epoch)
        writer.add_scalar('/test/TC', TC.avg,epoch)
            # plot progress
        print("\n| Validation Epoch #%d\t\tLoss: %.4f Share Mag@1: %.4f TC: %.4f" % (epoch,loss_avg.avg, share_mag.avg, TC.avg))
        print("| X: %.2f%% X-Xi: %.2f%% Xi: %.2f%% Xd: %.2f%%" %(top1.avg, top1_x_xi.avg, top1_xi.avg, top1_xd.avg))

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+epochs):
    start_time = time.time()

    train(epoch)
    scheduler.step()
    scheduler_c.step()
    if epoch % 10 == 9:
        test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
