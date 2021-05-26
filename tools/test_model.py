# -*- coding: utf-8 -*-
import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import pdb
import wandb
import sys
sys.path.append('.')

from tqdm import tqdm
from aa.networks import *
from advex.attacks import PGDAttack, NoAttack, DeltaAttack
from perceptual_advex.attacks import StAdvAttack
from utils.set import *

import robustness
from robustness.tools import folder, constants
from robustness.tools.helpers import get_label_mapping
from robustness.data_augmentation import Lighting

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def main(args):
    if args.dataset == 'cifar':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.expanduser("../data"), train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=32, shuffle=False)
        cudnn.benchmark = True

        model = Wide_ResNet(28, 10, 0.3, 10, True).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))

        vae = CVAE_cifar(d=32, z=2048, with_classifier=False)
        vae = nn.DataParallel(vae)
        save_model = torch.load(args.vae_path)
        model_dict = vae.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        vae.load_state_dict(model_dict)
        vae.cuda()
        vae.eval()

        normalize = CIFARNORMALIZE(32)
        innormalize = CIFARINNORMALIZE(32)

    else:
        label_map = get_subclass_label_mapping(ranges)
        crop_size = 224
        val_size = 256
        valdir = os.path.join(args.data, 'val')
        val_dataset = folder.ImageFolder(root=valdir,
                                         transform=transforms.Compose([
                                             transforms.Resize(val_size),
                                             transforms.CenterCrop(crop_size),
                                             transforms.ToTensor(),
                                         ]),
                                         label_mapping=label_map)
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True)
        model = resnet50(pretrained=True, num_classes=203, norm=True)
        model = nn.DataParallel(model)
        save_model = torch.load(args.model_path)['state_dict']
        model.load_state_dict(save_model)
        model.cuda()
        model.float()

        save_model = torch.load(args.vae_path)['state_dict']
        vae = CVAE_imagenet(d=64, k=128, with_classifier=False)
        vae = nn.DataParallel(vae)
        model_dict = vae.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        vae.load_state_dict(model_dict)
        vae.cuda()
        vae.float()

        normalize = IMAGENETNORMALIZE(224)
        innormalize = IMAGENETINNORMALIZE(224)

    model.eval(), vae.eval()
    model.apply(apply_dropout)
    validation_attacks = [
        NoAttack(),
        DeltaAttack(model, vae, eps_max=4/255, num_iterations=10)
        #PGDAttack(model, norm='l2', eps_max=1.0, num_iterations=100),
        #PGDAttack(model, num_iterations=100),
        #StAdvAttack(model, num_iterations=100),
    ]
    for attack in validation_attacks:
        adv_acc, ape_acc, n = 0, 0, 0
        attack_name = attack.__class__.__name__
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            x_adv = attack(x, t)
            y_adv = model(x_adv)
            pred_adv = y_adv.data.max(1)[1]
            adv_equal_flag = pred_adv.eq(t.data).cpu()
            adv_acc += adv_equal_flag.sum()

            y_ape = model(innormalize(vae(normalize(x_adv), noise=1))) + model(innormalize(vae(normalize(x_adv), noise=1.0))) + model(innormalize(vae(normalize(x_adv), noise=1.0)))
            pred_ape = y_ape.data.max(1)[1]
            ape_equal_flag = pred_ape.eq(t.data).cpu()
            ape_acc += ape_equal_flag.sum()
            n += t.size(0)
        print('attck {} adv {:.6f}, G(x) {:.6f}'.format(attack_name,
            adv_acc / n * 100,
            ape_acc / n * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/gpub/imagenet_raw")
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--model_path", type=str,
                        default="/gdata2/yangkw/auto_aug-master/results/cifar_baseline/wide_resnet.pth")
    parser.add_argument("--vae_path", type=str, default="/gdata2/yangkw/auto_aug-master/results/cifar_ce1/model_epoch172.pth")

    args = parser.parse_args()
    main(args)