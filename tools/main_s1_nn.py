from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

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
parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
parser.add_argument('--seed', default=666, type=int, help='seed')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--dim', default=8, type=int, help='CNN_embed_dim')
parser.add_argument('--re', default=0.5, type=float, help='reconstruction weight')
parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
alpha = 2.0
epochs = 300
learning_rate = 1.e-3
learning_rate_min = 2.e-4
ce_coef = 1.0
alpha = 2.0
args = parser.parse_args()
CNN_embed_dim = args.dim
re_coef = args.re
kl_coef = args.kl
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

# Model
print('\n[Phase 2] : Model setup')
model = VAE_s1(kernel_num=CNN_embed_dim, z_size=CNN_embed_dim)

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = AdamW([
                {'params': model.parameters()}
            ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * np.ceil(50000 / batch_size),
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
                _,_,_,_, xi,_, _ = model(X)

                grid_X = torchvision.utils.make_grid(X.data,  nrow=8, padding=2, normalize=True)
                writer.add_image( '_Batch_{batch}_{datasource}_X.jpg'.format(batch=batch_idx,datasource=datasource),grid_X,epoch)
                grid_Xi = torchvision.utils.make_grid(xi.data,  nrow=8, padding=2, normalize=True)
                writer.add_image('Batch_{batch}_{datasource}_Xi.jpg'.format( batch=batch_idx,datasource=datasource),grid_Xi,epoch)
                grid_X_Xi = torchvision.utils.make_grid((X-xi).data,  nrow=8, padding=2, normalize=True)
                writer.add_image( '_Batch_{batch}_{datasource}_X-Xi.jpg'.format( batch=batch_idx,datasource=datasource),grid_X_Xi,epoch)
                torchvision.utils.save_image(xi.data, os.path.join(save_path, 'Epoch_{epoch}_Batch_{batch}_{datasource}_xi.jpg'.format(epoch = epoch, batch = batch_idx, datasource = datasource)), nrow=8, padding=2, normalize=True)
                torchvision.utils.save_image((X-xi).data, os.path.join(save_path, 'Epoch_{epoch}_Batch_{batch}_{datasource}_X_xi.jpg'.format(epoch = epoch, batch = batch_idx, datasource = datasource)), nrow=8, padding=2, normalize=True)
                torchvision.utils.save_image(X.data, os.path.join(save_path, 'Epoch_{epoch}_Batch_{batch}_{datasource}_X.jpg'.format(epoch = epoch, batch = batch_idx, datasource = datasource)), nrow=8, padding=2, normalize=True)
    print('reconstruction complete!')

# Training
def train(epoch):

    model.train()
    model.training = True

    loss_avg = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()


    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=alpha)
        x, y, y_b = x.cuda(), y.cuda().view(-1, ), y_b.cuda().view(-1, )
        x, y = Variable(x), [Variable(y),Variable(y_b)]
        bs = x.size(0)
        optimizer.zero_grad()

        out, out1, out2, _, xi, mu, logvar = model(x)

        l1 = F.mse_loss(xi, x)
        entropy = (F.softmax(out1, dim=1) * F.log_softmax(out1, dim=1)).sum(dim=1).mean()
        cross_entropy = lam * F.cross_entropy(out2, y[0]) + (1. - lam) * F.cross_entropy(out2, y[1])
        l2 = cross_entropy + entropy
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * 1024
        loss = re_coef * l1 + ce_coef * l2 + kl_coef * l3
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5, correct, pred = accuracy(out.data, y[0].data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_entropy.update(entropy.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)
        top1.update(prec1.item(), bs)

        n_iter = epoch * len(trainloader) + batch_idx
        writer.add_scalar('loss', loss_avg.avg, n_iter)
        writer.add_scalar('loss_rec', loss_rec.avg, n_iter)
        writer.add_scalar('loss_ce', loss_ce.avg, n_iter)
        writer.add_scalar('loss_entropy', loss_entropy.avg, n_iter)
        writer.add_scalar('loss_kl', loss_kl.avg, n_iter)
        writer.add_scalar('acc', top1.avg, n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Acc@1: %.3f%%'
                    %(epoch, epochs, batch_idx+1,
                       len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg, loss_kl.avg, top1.avg))
            #sys.stdout.flush()
    if epoch % 10 == 1 :
        torch.save(model.state_dict(),
                   os.path.join(save_path, 'model_new.pth'))  # save motion_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(save_path, 'optimizer_new.pth'))  # save optimizer

        print("Epoch {} model saved!".format(epoch + 1))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, train=True, model=model, save_path=save_path)
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, train=False, model=model, save_path=save_path)


def test(epoch):
    # set model as testing mode
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

            out, out_i, out_x_xi, hi, xi, mu, logvar = model(x)
            loss = F.mse_loss(xi, x)
            test_loss += loss.item()  # sum up batch loss
            loss_avg.update(loss.data.item(), bs)
            # measure accuracy and record loss
            prec1, _,_,_ = accuracy(out.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

            prec1_x_xi, _,_,_ = accuracy(out_x_xi.data, y.data, topk=(1, 5))
            top1_x_xi.update(prec1_x_xi.item(), bs)

            prec1_xi,  _,_,_ = accuracy(out_i.data, y.data, topk=(1, 5))
            top1_xi.update(prec1_xi.item(), bs)

            tc = total_correlation(hi, mu, logvar)/  bs / CNN_embed_dim
            TC.update(tc.item(), bs)

        writer.add_scalar('/test/loss', loss_avg.avg, epoch)
        writer.add_scalar('/test/X-acc', top1.avg, epoch)
        writer.add_scalar('/test/X-Xi-acc', top1_x_xi.avg,epoch)
        writer.add_scalar('/test/Xi-acc', top1_xi.avg,epoch)
        writer.add_scalar('/test/TC', TC.avg,epoch)
            # plot progress
        print("\n| Validation Epoch #%d\t\tLoss: %.4f TC: %.4f" % (epoch,loss_avg.avg, TC.avg))
        print("| X: %.2f%% X-Xi: %.2f%% Xi: %.2f%%" %(top1.avg, top1_x_xi.avg, top1_xi.avg))

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+epochs):
    start_time = time.time()
    test(epoch)
    train(epoch)
    scheduler.step()
    if epoch % 10 == 9:
        test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
