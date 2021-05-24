import argparse
import logging
import os
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torchvision import datasets, transforms
import wandb
import sys
sys.path.append('.')
from utils.set import *
import robustness
from robustness.tools import folder, constants
from robustness.tools.helpers import get_label_mapping
from robustness.data_augmentation import Lighting
from aa.networks import *
from advex.attacks import *

logger = logging.getLogger(__name__)

imagenet_mean = (0.4914, 0.4822, 0.4465)
imagenet_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
std = torch.tensor(imagenet_std).view(3,1,1).cuda()

def evaluate_attack(test_loader, model, attack):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            Adv_X = attack(X, y)
            output = model(Adv_X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data', default='/gpub/imagenet_raw', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datasets', default='simagenet', type=str)
    parser.add_argument('--norm', default='l2', type=str)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='./results/train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='dynamic', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

def main():
    args = get_args()
    wandb.init(config=args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.datasets =='rimagenet':
        label_map = get_label_mapping('restricted_imagenet',
                                     constants.RESTRICTED_IMAGNET_RANGES)
        num_classes = 9
    elif args.datasets =='simagenet':
        label_map = get_subclass_label_mapping(ranges)
        num_classes = 203

    # create model
    model = resnet50(pretrained=True, num_classes=num_classes).cuda()
    model.train()

    crop_size = 224
    val_size = 256

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = folder.ImageFolder(root=traindir,
            transform=transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
            label_mapping=label_map)

    val_dataset = folder.ImageFolder(root=valdir,
            transform=transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
         label_mapping=label_map)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True )

    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    args.lr_max = args.lr_max*float(args.batch_size)/256.
    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=args.lr_max, betas=(0.9, 0.999), weight_decay=1.e-6)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 3, lr_steps * 2 / 3], gamma=0.1)
    attack = PGDAttack(model, optimizer, num_iterations=5)
    #attack = nn.DataParallel(attack)
    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            model.eval()
            orig_logits = model(X)
            to_attack = orig_logits.argmax(1) == y
            Adv_X = X.clone()
            if to_attack.sum()>0:
                Adv_X[to_attack] = attack(X[to_attack], y[to_attack])
            model.train()
            output = model(Adv_X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        print('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        wandb.log({"epoch": epoch, "train/Prec@1": train_acc/train_n,
                 "train/loss": train_loss/train_n,  "lr": lr,
                 })
        if epoch % 10 == 0:
            train_time = time.time()
            torch.save(model.module.state_dict(), os.path.join(args.out_dir, 'model_epoch{}.pth'.format(epoch)))
            logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

            # Evaluation
            model_test = resnet50(pretrained=True, num_classes=num_classes).cuda()
            model_test = torch.nn.DataParallel(model_test)
            model_test.load_state_dict(model.state_dict())
            model_test.float()
            model_test.eval()

            noattack = NoAttack()
            pgdl2 = PGDAttack(model_test, eps_max=1.0, norm='l2', num_iterations=100)
            pgdlinf =  PGDAttack(model_test, num_iterations=100)
            pgd_loss_l2, pgd_acc_l2 = evaluate_attack(test_loader, model_test, pgdl2)
            pgd_loss, pgd_acc = evaluate_attack(test_loader, model_test, pgdlinf)
            test_loss, test_acc = evaluate_attack(test_loader, model_test, noattack)

            logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD_L2 Loss \t PGD_L2 Acc')
            logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc, pgd_loss_l2, pgd_acc_l2)
            print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD_L2 Loss \t PGD_L2 Acc')
            print('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc, pgd_loss_l2, pgd_acc_l2)
            wandb.log({"Val/Noattack@1": test_acc,
                       "Val/PGDL2@1": pgd_acc_l2,
                       "Val/PGD@1": pgd_acc,
                 }, commit=False)

if __name__ == "__main__":
    main()
