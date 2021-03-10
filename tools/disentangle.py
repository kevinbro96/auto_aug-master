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
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

def get_cifar_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class CIFARNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class CIFARINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(self.mean)
        return x

dim=64
batch_size=10
vae_dir = 'results/emb64_301/model_new.pth'
dataset = 'cifar10'
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

vae = CVAE_s2(d=8, z=dim)
vae = nn.DataParallel(vae)
save_model = torch.load(vae_dir)
model_dict = vae.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

model_dict.update(state_dict)
vae.load_state_dict(model_dict)
vae.eval()
wandb.init()
cifarinnormlize = CIFARINNORMALIZE(32)
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
                grid_image = torchvision.utils.make_grid(x.data, nrow=1, padding=2, normalize=True)
                images.append(grid_image)
                grid_image_xi = torchvision.utils.make_grid(xi.data, nrow=1, padding=2, normalize=True)
                images_xi.append(grid_image_xi)
                for step in range(-5,5):
                    hnew = hi.clone()
                    hnew[:,d] = hnew[:,d] + step * 0.3
                    xinew = vae(hnew, mode='hi-xi')
                    xnew = x- xi+xinew
                    grid_image = torchvision.utils.make_grid(xnew.data, nrow=1, padding=2, normalize=True)
                    images.append(grid_image)
                    grid_image_xi = torchvision.utils.make_grid(xinew.data, nrow=1, padding=2, normalize=True)
                    images_xi.append(grid_image_xi)
                wandb.log({f'x/dim-{str(d)}': [
                    wandb.Image(torch.cat(images, dim=2))]})
                wandb.log({f'xi/dim-{str(d)}': [
                    wandb.Image(torch.cat(images_xi, dim=2))]})





