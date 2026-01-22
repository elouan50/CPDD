import os
from pydantic import BaseModel
from typing import Annotated, Literal
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import random
import shutil
from misc.utils import Logger
import torch.backends.cudnn as cudnn
import json

class Synthesizer():
    """Condensed data class
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc *
                               (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(
                            img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained,
                        0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(
                    img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(
                img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(
                    data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(
                    data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(
                augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(
                augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(
                augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(
                augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(
            data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        result = test_data(args, loader, val_loader,
                           test_resnet=False, logger=logger)

        return result


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(
            mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10



    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    # width check
    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]

    return train_dataset, val_loader


def tune_lr_img(args, lr_img):
    # Use mse loss for 32x32 img and ConvNet
    ipc_base = 10
    if args.dataset == 'imagenet':
        imsize_base = 224
    elif args.dataset == 'speech':
        imsize_base = 64
    elif args.dataset == 'mnist':
        imsize_base = 28
    else:
        imsize_base = 32

    param_ratio = (args.ipc / ipc_base)
    if args.size > 0:
        param_ratio *= (args.size / imsize_base)**2

    lr_img = lr_img * param_ratio
    return lr_img


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(
        mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, model, h_p=None):
    loss = None
    k = img_real.shape[0]
    with torch.no_grad():
        feat_tg, _ = model.get_feature(
            img_real[:k], args.idx_from, args.idx_to)
    feat, _ = model.get_feature(img_syn, args.idx_from, args.idx_to)

    proto_loss = add_loss(loss, dist(
        feat_tg[len(feat_tg)-1].mean(0), feat[len(feat)-1].mean(0), method=args.metric))

    proto_tg = feat_tg[len(feat)-1].mean(0)
    proto_tg = proto_tg.view(proto_tg.shape[0], -1)
    proto_tg = proto_tg.reshape(-1)
    feat_tg_view = feat_tg[len(feat_tg)-1].view(feat_tg[len(feat_tg)-1].size(0), -1)   

    proto_syn = feat[len(feat)-1].mean(0)
    proto_syn = proto_syn.view(proto_syn.shape[0], -1)
    proto_syn = proto_syn.reshape(-1)
    feat_view = feat[len(feat) - 1].view(feat[len(feat) - 1].size(0), -1)

    centered_real = feat_tg_view - proto_tg
    centered_syn = feat_view - proto_syn

    cov_real = torch.matmul(centered_real.t(), centered_real) / (feat_tg_view.size(0) - 1)
    cov_syn = torch.matmul(centered_syn.t(), centered_syn) / (feat_view.size(0) - 1)
    semantic_loss = torch.mul(dist(cov_syn, cov_real, method=args.metric)/proto_syn.shape[0], args.cov_weight)
    loss = add_loss(proto_loss, semantic_loss)
    
    if h_p is not None:
        h_p_loss = torch.mul(dist(feat[len(feat)-1].mean(0), h_p, method=args.metric)/proto_syn.shape[0], args.h_p_weight)
        loss = add_loss(loss, h_p_loss)

    return loss

def pickle_data(args, synset):
    """Save condensed data as a pickle file
    """
    images = synset.data.detach().cpu()

    batch = {
        b'data': images.numpy().reshape(len(images), -1),
        b'batch_label': f'distilled_batch_{args.cpdd_iter}'
    }

    with open(os.path.join(args.save_dir, f"distilled_data_batch_{args.cpdd_iter}"), "wb") as f:
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init_data)
    # save_img(os.path.join(args.save_dir, 'init.png'),
    #          synset.data,
    #          unnormalize=False,
    #          dataname=args.dataset)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    # save_img(os.path.join(args.save_dir, f'aug.png'),
    #          aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
    #          unnormalize=True,
    #          dataname=args.dataset)

    # if not args.test:
    #     synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(
        synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
   
    it_log = 20
    it_test = [i for i in range(0, args.niter+1, args.evaluate_iter)]

    logger(
        f"\nStart condensing with {args.match} matching for {args.niter} iteration")

    best_acc = -1

    smooth_syns = [None] * nclass 
    h_p = [None] * nclass

    for it in range(args.niter):
        j = random.randint(0, args.pretrained_model_number-1)
        model = define_model(args, nclass).to(device)
        if args.dataset == 'cifar10':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'cifar100':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'svhn':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'mnist':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'fashion':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        ts.set()

        for c in range(nclass):
            img, lab = loader_real.class_sample(c)
            img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
            ts.stamp("data")
            n = img.shape[0]
            img_aug = aug(torch.cat([img, img_syn]))
            ts.stamp("aug")

            optim_img.zero_grad()
            if it > args.smooth_iter:
                loss = matchloss(args, img_aug[:n], img_aug[n:], model, h_p=h_p[c])
            else:
                loss = matchloss(args, img_aug[:n], img_aug[n:], model)
            loss_total += loss.item()
            ts.stamp("loss")
            loss.backward()
            optim_img.step()
            ts.stamp("backward")
            ts.flush()
     
        for c in range(nclass):
            img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
            ts.stamp("data")
            syn_img_aug = aug(torch.cat([img_syn]))
            ts.stamp("aug")
            if (it == 0):
                with torch.no_grad():
                    feature_h, _ = model.get_feature(syn_img_aug, args.idx_from, args.idx_to)
                smooth_syns[c] = feature_h[len(feature_h)-1].mean(0)
                h_p[c] = smooth_syns[c]
            else:
                with torch.no_grad():
                    feature_h, _ = model.get_feature(syn_img_aug, args.idx_from, args.idx_to)
                smooth_syns[c] = feature_h[len(feature_h)-1].mean(0)
                h_p[c] = (1 -args.smooth_factor) * smooth_syns[c] + args.smooth_factor * h_p[c]

         # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass:.1f}")
        if (it + 1) in it_test:
            if not args.test:
                conv_result = synset.test(args, val_loader, logger)
                if conv_result > best_acc:
                    best_acc = conv_result
                    
                    pickle_data(args, synset)
                    # torch.save(
                    #     [synset.data.detach().cpu(), synset.targets.cpu()],
                    #     os.path.join(args.save_dir, f'data_best.pt'))
                    # print("best img and data updated!")
                    # save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                    #          synset.data,
                    #          unnormalize=False,
                    #          dataname=args.dataset)
                logger(synset.data.shape)
                logger(
                    "->->->->->->->->->->->->-> Best Result: {:.1f}".format(best_acc))

class Distillation(BaseModel):
    # Dataset
    dataset: Annotated[Literal["mnist", "fashion", "svhn", "cifar10", "cifar100"], "dataset (options: mnist, fashion, svhn, cifar10, cifar100)"] = 'cifar10'
    data_dir: Annotated[str, "directory that containing dataset"] = './data'
    imagenet_dir: Annotated[str, "path to imagenet dataset"] = '/distillation/ssd_data/imagenet/'
    nclass: Annotated[int, "number of classes in training dataset"] = 10
    dseed: Annotated[int, "seed for class sampling"] = 0
    size: Annotated[int, "spatial size of image"] = 224
    phase: Annotated[int, "index for multi-processing"] = -1
    nclass_sub: Annotated[int, "number of classes for each process"] = -1
    load_memory: Annotated[bool, "load training images on the memory"] = True
    
    # Network
    net_type: Annotated[Literal["convnet", "resnet", "resnet_ap"], "network type: resnet, resnet_ap, convnet"] = 'convnet'
    norm_type: Annotated[Literal["batch", "instance", "sn", "none"], "normalization type"] = 'instance'
    depth: Annotated[int, "depth of the network"] = 10
    width: Annotated[float, "width of the network"] = 1.0
    
    # Training
    pretrained_model_number: Annotated[int, "number of pre-trained models"] = 10
    pretrained_epochs: Annotated[int, "number of pre-trained epochs"] = 20
    batch_size: Annotated[int, "mini-batch size for training"] = 64
    lr: Annotated[float, "initial learning rate"] = 0.01
    momentum: Annotated[float, "momentum"] = 0.9
    weight_decay: Annotated[float, "weight decay"] = 5e-4
    seed: Annotated[int, "random seed for training"] = 0
    pretrained: Annotated[bool, "use pretrained model"] = False
    save_pretrain_dir: Annotated[str, "directory that saving pre trained model"] = './distillation/pre_trained_model'
    
    # Mixup
    mixup: Annotated[Literal["vanilla", "cut"], "mixup choice for evaluation"] = 'cut'
    mixup_net: Annotated[Literal["vanilla", "cut"], "mixup choice for training networks in condensation stage"] = 'cut'
    beta: Annotated[float, "mixup beta distribution"] = 1.0
    mix_p: Annotated[float, "mixup probability"] = 0.5
    
    # Logging
    cpdd_iter: Annotated[int, "checkpoint iteration for distilled data"] = 10000
    print_freq: Annotated[int, "print frequency"] = 10
    verbose: Annotated[bool, "to print the status at every iteration"] = False
    workers: Annotated[int, "number of data loading workers"] = 8
    save_ckpt: Annotated[bool, "save checkpoint"] = False
    tag: Annotated[str, "name of experiment"] = ''
    test: Annotated[bool, "for debugging, do not save results"] = False
    time: Annotated[bool, "measuring time for each step"] = False
    
    # Condense
    cov_weight: Annotated[float, "semantic weight"] = 50.0
    h_p_weight: Annotated[float, "historical prototype weight"] = 0.2
    smooth_factor: Annotated[float, "smoothing factor"] = 0.99
    epochs: Annotated[int, "number of test epochs"] = 1500
    ipc: Annotated[int, "number of condensed data per class"] = -1
    factor: Annotated[int, "multi-formation factor (1 for IDC-I)"] = 1
    decode_type: Annotated[Literal["single", "multi", "bound"], "multi-formation type"] = 'single'
    init_data: Annotated[Literal["random", "noise", "mix"], "condensed data initialization type"] = 'random'
    aug_type: Annotated[str, "augmentation strategy for condensation matching objective"] = 'color_crop_cutout'
    
    # Matching objective
    match: Annotated[Literal["feat", "grad", "semantic"], "feature or gradient matching"] = 'grad'
    metric: Annotated[Literal["mse", "l1", "l1_mean", "l2", "cos"], "matching objective"] = 'l1'
    bias: Annotated[bool, "match bias or not"] = False
    fc: Annotated[bool, "match fc layer or not"] = False
    f_idx: Annotated[str, "feature matching layer (comma separation)"] = '4'
    
    # Optimization
    niter: Annotated[int, "number of outer iteration"] = 10000
    smooth_iter: Annotated[int, "number of starting smooth iteration"] = 2000
    evaluate_iter: Annotated[int, "number of outer iteration evaluating the performance of distilled data"] = 100
    batch_real: Annotated[int, "batch size of real training data used for matching"] = 256
    batch_syn_max: Annotated[int, "maximum number of synthetic data used for each matching"] = 256
    lr_img: Annotated[float, "condensed data learning rate"] = 5e-3
    mom_img: Annotated[float, "condensed data momentum"] = 0.5
    reproduce: Annotated[bool, "for reproduce our setting"] = False
    
    # Test
    slct_type: Annotated[str, "selection type"] = 'DSDM'
    repeat: Annotated[int, "number of test repetition"] = 1
    dsa: Annotated[bool, "use DSA augmentation for evaluation or not"] = True
    dsa_strategy: Annotated[str, "DSA strategy"] = 'color_crop_cutout_flip_scale_rotate'
    rrc: Annotated[bool, "use random resize crop for ImageNet"] = True
    same_compute: Annotated[bool, "match evaluation training steps for IDC"] = False
    name: Annotated[str, "name of the test data folder"] = ''
    
    # Derived attributes
    datatag: Annotated[str, "dataset tag"] = ''
    modeltag: Annotated[str, "model tag"] = ''
    save_dir: Annotated[str, "directory that saving results"] = ''
    epoch_print_freq: Annotated[int, "print frequency during evaluation"] = 1
    idx_from: Annotated[int, "feature matching from layer"] = 0
    idx_to: Annotated[int, "feature matching to layer"] = -1
    augment: Annotated[bool, "use augmentation for evaluation"] = True
    nch: Annotated[int, "number of channels in image"] = 3
    
    def init(self):
        """Apply dataset-specific configurations and derive attributes"""

        # Dataset-specific configurations
        if self.dataset[:5] == 'cifar':
            self.size = 32
            self.mix_p = 0.5
            self.dsa = True
            if self.dataset == 'cifar10':
                self.nclass = 10
            elif self.dataset == 'cifar100':
                self.nclass = 100
        
        elif self.dataset == 'svhn':
            self.size = 32
            self.nclass = 10
            self.mix_p = 0.5
            self.dsa = True
            self.dsa_strategy = remove_aug(self.dsa_strategy, 'flip')
        
        elif self.dataset[:5] == 'mnist':
            self.nclass = 10
            self.size = 28
            self.nch = 1
            self.mix_p = 0.5
            self.dsa = True
            self.dsa_strategy = remove_aug(self.dsa_strategy, 'flip')
        
        elif self.dataset == 'fashion':
            self.nclass = 10
            self.size = 28
            self.nch = 1
            self.mix_p = 0.5
            self.dsa = True
        
        elif self.dataset == 'speech':
            self.nch = 1
            self.size = 64
            if self.net_type == 'convnet':
                self.depth = 4
            self.nclass = 8
            self.mixup = 'vanilla'
            self.mixup_net = 'vanilla'
            self.dsa = False
        
        self.datatag = f'{self.dataset}'
        
        # Network-specific configurations
        if self.net_type == 'convnet':
            if self.depth > 4:
                self.depth = 3
            self.f_idx = str(self.depth - 1)
        
        self.modeltag = f'{self.net_type}{self.depth}'
        if self.net_type == 'resnet_ap':
            self.modeltag = f'resnet{self.depth}ap'
        if self.net_type == 'convnet':
            self.modeltag = f'conv{self.depth}'
        if self.norm_type == 'instance':
            self.modeltag += 'in'
        if self.width != 1.0:
            self.modeltag += f'_w{self.width}'
        
        # Default initialization for multi-formation
        if self.factor > 1:
            self.init_data = 'mix'
        
        # Build experiment tag
        if self.tag != '':
            self.tag = f'_{self.tag}'
        
        if self.ipc > 0:
            if self.slct_type == 'random':
                self.tag += f'_rand{self.ipc}'
            
            elif self.slct_type == 'DSDM':
                self.tag += f'_semantic'
                f_list = [int(s) for s in self.f_idx.split(',')]
                if len(f_list) == 1:
                    f_list.append(-1)
                self.idx_from, self.idx_to = f_list
                self.metric = 'mse'
                
                self.tag += f'_{self.metric}'
                if self.mixup_net == 'cut':
                    self.tag += f'_cut'
                if self.lr != 0.01:
                    self.tag += f'_nlr{self.lr}'
                if self.weight_decay != 5e-4:
                    self.tag += f'_wd{self.weight_decay}'
                
                if self.factor > 0:
                    self.tag += f'_factor{self.factor}'
                    if self.decode_type != 'single':
                        self.tag += f'_{self.decode_type}'
                if self.aug_type != 'color_crop_cutout':
                    self.tag += f'_{self.aug_type}'
                
                self.tag += f'_lr{self.lr_img}'
                self.lr_img = tune_lr_img(self, self.lr_img)
                if self.momentum != 0.9:
                    self.tag += f'_mom{self.momentum}'
                if self.batch_real != 64:
                    self.tag += f'_b_real{self.batch_real}'
                if self.batch_syn_max != 128:
                    self.tag += f'_synmax{self.batch_syn_max}'
                
                self.tag += f'_{self.init_data}'
                self.tag += f'_ipc{self.ipc}'
                
                if self.nclass_sub > 0:
                    self.tag += f'_{self.nclass_sub}'
                if self.phase >= 0:
                    self.tag += f'_phase{self.phase}'
        else:
            if self.mixup != 'vanilla':
                self.tag += f'_{self.mixup}'
        
        # Result folder name
        if self.test:
            self.save_dir = './distillation/results/test'
        else:
            self.save_dir = f'./data/{self.dataset}' # f"./distillation/results/{self.datatag}/{self.modeltag}{self.tag}"
        
        # Evaluation setting
        if self.ipc > 0:
            self.epochs = 1500
            self.epoch_print_freq = self.epochs
        else:
            self.epoch_print_freq = 1
        
        # Augmentation setting
        if self.mixup == 'cut':
            self.dsa_strategy = remove_aug(self.dsa_strategy, 'cutout')
        if self.dsa:
            self.augment = False
        else:
            self.augment = True
        
    def main(self):
        """Main function to run distillation"""
        self.init()
        
        assert self.ipc > 0

        cudnn.benchmark = True
        if self.seed > 0:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        os.makedirs(self.save_dir, exist_ok=True)

        logger = Logger(self.save_dir)
        logger(f"Save dir: {self.save_dir}")

        condense(self, logger)

if __name__ == '__main__':

    distillation = Distillation(ipc=10, factor=2, dataset='cifar10', niter=1000)
    distillation.main()
