import argparse
import os
import shutil
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pdb
import numpy as np

import wandb
import sys
sys.path.append('.')
from utils.set import *
import robustness
from robustness.tools import folder, constants
from robustness.tools.helpers import get_label_mapping
from robustness.data_augmentation import Lighting
from aa.networks import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--save_dir', default='./results/imagenet/', type=str, help='save_dir')
    parser.add_argument('--datasets', default='simagenet', type=str, help='dataset')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)

    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')
    parser.add_argument('--re', nargs='+', type=int)
    parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
    parser.add_argument('--ce', default=1.0, type=float, help='cross entropy weight')
    args = parser.parse_args()
    return args

def main():
    global best_prec1, args
    args = parse()
    setup_logger(args.save_dir)
    run = setup_run(args)
    is_master = args.local_rank == 0
    learning_rate_min = 2.e-4
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    # Data loading code
    if args.datasets =='rimagenet':
        label_map = get_label_mapping('restricted_imagenet',
                                     constants.RESTRICTED_IMAGNET_RANGES)
        num_classes = 9
    elif args.datasets =='simagenet':
        label_map = get_subclass_label_mapping(ranges)
        num_classes = 203

    # create model
    model = CVAE_imagenet(d=64, k=128, num_classes=num_classes)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=1.e-6)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)
    if is_master:
        run.watch(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    train_dataset = folder.ImageFolder(root=traindir,
            transform=transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            # normalize,
        ]),
            label_mapping=label_map)
    n_train = len(train_dataset)
    val_dataset = folder.ImageFolder(root=valdir,
            transform=transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
            #transforms.ToTensor(),
            ]),
         label_mapping=label_map)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15,
                                                           eta_min=learning_rate_min)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, run)
        scheduler.step()
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, run)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch)
            if epoch % 10 == 1:
                reconst_images(run, dataloader=val_loader, model=model)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def train(train_loader, model, criterion, optimizer, epoch, run):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    if epoch < 30:
        re = args.re[0]
    elif epoch < 60:
        re = args.re[1]
    else:
        re = args.re[2]

    prefetcher = data_prefetcher(train_loader)
    x, y = prefetcher.next()
    i = 0
    while x is not None:
        i += 1
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        x, y, y_b, lam, _ = mixup_data(x, y, alpha=args.alpha)
        bs = x.size(0)
        out, out1, out2,  xi, z_e, emb = model(x)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        l1 = F.mse_loss(xi, x)
        entropy = (F.softmax(out1, dim=1) * F.log_softmax(out1, dim=1)).sum(dim=1).mean()
        cross_entropy = lam * F.cross_entropy(out2, y) + (1. - lam) * F.cross_entropy(out2, y_b)
        l2 = cross_entropy + entropy
        l3 = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1)) + 0.5*torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))
        loss = re * l1 + args.ce * l2 + args.kl * l3

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5, _, _ = accuracy(out2.data, y, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
                l1 = reduce_tensor(l1.data)
                entropy = reduce_tensor(entropy.data)
                cross_entropy = reduce_tensor(cross_entropy.data)
                l3 = reduce_tensor(l3.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), bs)
            top1.update(to_python_float(prec1), bs)
            top5.update(to_python_float(prec5), bs)
            loss_rec.update(to_python_float(l1), bs)
            loss_ce.update(to_python_float(cross_entropy), bs)
            loss_entropy.update(to_python_float(entropy), bs)
            loss_kl.update(to_python_float(l3), bs)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'loss_rec {loss_rec.avg:.7f}\t'
                      'loss_ce {loss_ce.avg:.7f}\t'
                      'loss_entropy {loss_entropy.avg:.3f}\t'
                      'loss_kl {loss_kl.avg:.3f}'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5,
                       loss_rec=loss_rec, loss_ce=loss_ce,
                       loss_entropy=loss_entropy, loss_kl=loss_kl))
                run.log({"Loss":losses.avg, "Prec@1":top1.avg,
                         "Loss_rec":loss_rec.avg, "loss_ce":loss_ce.avg,
                         "loss_entropy":loss_entropy.avg, "loss_kl":loss_kl.avg,
                         're_weight': re, "learning rate":optimizer.param_groups[0]['lr']})
        if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        x, y = prefetcher.next()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

def reconst_images(run, batch_size=64, batch_num=2, dataloader=None, model=None):
    cifar10_dataloader = dataloader

    model.eval()

    prefetcher = data_prefetcher(cifar10_dataloader)
    X, y = prefetcher.next()
    i = 0
    while X is not None:
        i += 1
        with torch.no_grad():
            if i >= batch_num:
                break
            else:
                _,_,_, xi, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                run.log({"_Batch_{batch}_X.jpg".format(batch=i): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(xi[:batch_size].data, nrow=8, padding=2, normalize=True)
                run.log({"_Batch_{batch}_Xi.jpg".format(batch=i): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - xi[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                run.log({"_Batch_{batch}_X-Xi.jpg".format(batch=i): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
        X, y = prefetcher.next()
    print('reconstruction complete!')

def validate(val_loader, model, criterion, run):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc_avg = AverageMeter()
    sparse_avg = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    x, y = prefetcher.next()
    i = 0
    while x is not None:
        i += 1

        # compute output
        with torch.no_grad():
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(bs, -1)),p=2,dim=1)
            out, out_i, out_x_xi,  xi, mu, logvar = model(x)
            acc_xi = 1-F.mse_loss(torch.div(xi,norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                              torch.div(x,norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                              reduction = 'sum')/bs
            acc_xd = 1-F.mse_loss(torch.div(x-xi,norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                              torch.div(x,norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                              reduction = 'sum')/bs

        # measure accuracy and record loss
        prec1, prec5, _, _ = accuracy(out_x_xi.data, y, topk=(1, 5))

        if args.distributed:
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            acc_xi = reduce_tensor(acc_xi)
            acc_xd = reduce_tensor(acc_xd)

        top1.update(to_python_float(prec1), bs)
        top5.update(to_python_float(prec5), bs)
        acc_avg.update(to_python_float(acc_xi), bs)
        sparse_avg.update(to_python_float(acc_xd), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Acc_xi {acc_avg.avg:.3f}\t'
                  'Acc_xd {sparse_avg.avg:.3f}'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5,
                   acc_avg=acc_avg, sparse_avg=sparse_avg))

        x, y = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if args.local_rank == 0:
        run.log({"Val/Prec@1":top1.avg, "Val/Prec@5":top5.avg,
                 "Val/Acc_xi": acc_avg.avg,  "Val/Acc_xd": sparse_avg.avg,
                 }, commit=False)
    return top1.avg


def save_checkpoint(state, is_best, epoch):
    filename = os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
