import torchvision
import torchvision.transforms as transforms
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys
sys.path.append('.')
from aa.networks import *
import aa.config as cf
import argparse

parser = argparse.ArgumentParser(description='Disentanglement')
parser.add_argument('--norm', default=1.0, type=float, help='norm')
parser.add_argument('--vae_dir', type=str, default='results/emb64_301/model_new.pth')
args = parser.parse_args()
dim=64
batch_size=10
dataset = 'cifar10'
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

vae = CVAE_s2(d=8, z=dim)
vae = nn.DataParallel(vae)
save_model = torch.load(args.vae_dir)
model_dict = vae.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

model_dict.update(state_dict)
vae.load_state_dict(model_dict)
vae.eval()
wandb.init(config=args)


with torch.no_grad():
    for batch_idx, (x, y) in enumerate(testloader):
        if batch_idx >= 1:
            break
        else:
            x, y = x.cuda(), y.cuda().view(-1, )
            hi = vae(x, mode='x-hi')
            xi = vae(hi, mode='hi-xi')
            for d in range(dim):
                images=[]
                images_xi = []
                for step in range(-5,6):
                    hnew = hi.clone()
                    hnew[:,d] = hnew[:,d] + step * args.norm
                    xinew = vae(hnew, mode='hi-xi')
                    xnew = x- xi+xinew
                    grid_image = torchvision.utils.make_grid(xnew.data, nrow=1, padding=2, normalize=True)
                    images.append(grid_image)
                    grid_image_xi = torchvision.utils.make_grid(xinew.data, nrow=1, padding=2, normalize=True)
                    images_xi.append(grid_image_xi)
                wandb.log({f'x/dim-{str(d)}': [
                    wandb.Image(torch.cat(images, dim=2))]}, commit=False)
                wandb.log({f'xi/dim-{str(d)}': [
                    wandb.Image(torch.cat(images_xi, dim=2))]}, commit=False)





