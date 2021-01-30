import os
import sys
import math
import fire
import json
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch.optim import Adam
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from unet_stylegan2.diff_augment import DiffAugment

######### LostGAN related imports ##############
from .mask_regression import *
from .norm_module import *
from .sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.ops import RoIAlign
BatchNorm = SynchronizedBatchNorm2d
from .sync_batchnorm import DataParallelWithCallback
from data.cocostuff_loader import *
from utils.logger import setup_logger
from utils.util import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data.mnist_loader_v2 import MNISTDataset
################################################


assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png', 'webp']
EPS = 1e-8

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# helpers

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images,
                           grad_outputs=list(map(lambda t: torch.ones(t.size()).cuda(), outputs)),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam

def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source

def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target

# dataset

def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, num_channels):
        self.num_channels = num_channels
    def __call__(self, tensor):
        return tensor.expand(self.num_channels, -1, -1)

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(num_channels))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

def random_float(lo, hi):
    return lo + (hi - lo) * random.random()

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random.random() * delta)
    w_delta = int(random.random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size, types):
        super().__init__()
        self.D = D
        self.types = types

    def forward(self, images, bbox, label, prob = 0.1, detach = False):
        flip = False
        if random.random() < prob:
            flip = True
            images = random_hflip(images, prob=0.5)
            bbox_ = bbox.clone()
            bbox_[:, :, 0] += 2 * (0.5 - bbox_[:, :, 0])
            bbox_[:, :, 0] -= bbox_[:, :, 2]
            
            if self.types != []:
                images = DiffAugment(images, types=self.types)

        if detach:
            images.detach_()
        
        if flip == True:
            return self.D(images, bbox_, label), images
        elif flip == False:
            return self.D(images, bbox, label), images

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, num_w = 128, upsample = True, upsample_rgb = True, rgba = False, predict_mask = True, psp_module = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)
        
        ######### LostGAN related components#################
        
        self.b1 = SpatialAdaptiveSynBatchNorm2d(filters, num_w = num_w, batchnorm_func = BatchNorm)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(filters, num_w = num_w, batchnorm_func = BatchNorm)
        
        
        self.predict_mask = predict_mask
        
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(filters, 100),
                                               nn.Conv2d(100, 184, kernel_size = 1))
            
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(filters, 100, 3, 1, 1),
                                               BatchNorm(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, 184, 1, 1, 0, bias = True))
        
        #######################################################
        

    def forward(self, x, prev_rgb, istyle, inoise, w, bbox):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x_ = x + noise1
        
        ############ISLA#################
        x = self.b1(x_, w, bbox)
        #################################
        
        x = self.activation(x)
        
        

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x_ = x + noise2
        
        ###########ISLA##################
        x = self.b2(x_, w, bbox)
        #################################
        
        x = self.activation(x)
        
        if self.predict_mask:
            mask = self.conv_mask(x)
        else:
            mask = None

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb, mask

def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )

class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res

class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride = 2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x
    
class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan)
            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        x = self.initial_conv(x)
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out

############################### LostGAN version of generator and discriminator#######################################################

############################## Lost Generator ######################################################################################

class Lost_Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, no_const = False, fmap_max = 512, z_dim = 128, num_classes = 10, output_dim = 3):
        super().__init__()
        
        num_w = 128+180
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        
        predict_mask = True
        psp_module = False
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan)
            self.attns.append(attn_fn)
            
            if ind == self.num_layers - 1:
                predict_mask = False
            
            if ind == self.num_layers - 2:
                psp_module = True
            
            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent,
                predict_mask = predict_mask,
                psp_module = psp_module,
                num_w = num_w
            )
            self.blocks.append(block)
            
            psp_module = False
        
        ###################### LostGAN related settings ###################################
        
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, 180)
        num_w = 128 + 180
        
        self.sigmoid = nn.Sigmoid()
        self.mask_regress = MaskRegressNetv2(num_w)
        
        
        for k in range(self.num_layers - 1):
            temp = nn.Parameter(torch.zeros(1, self.num_classes, 1))
            self.__setattr__('alpha_%s' % (str(k)), temp)
            
        
        # mapping function
        mapping = list()
        
        self.mapping = nn.Sequential(*mapping)
        
                                
        ###################################################################################
        
    # def __getattr__(self, name):
        # return getattr(self.module, name)
    
    
    def forward(self, styles, input_noise, z_obj_style, bbox, y = None):
        
        batch_size, num_o = z_obj_style.size(0), z_obj_style.size(1)
        label_embedding = self.label_embedding(y)
        
        z_obj_style = z_obj_style.view(batch_size * num_o, -1)
        label_embedding = label_embedding.view(batch_size * num_o, -1)
        
        latent_vector = torch.cat((z_obj_style, label_embedding), dim = 1).view(batch_size, num_o, -1)
        
        w = self.mapping(latent_vector.view(batch_size * num_o, -1))
    
        
        # preprocess bbox
        bmask = self.mask_regress(w, bbox)
        bbox_mask_ = bbox_mask(z_obj_style, bbox, 64, 64)
        
        
        
        # batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        x = self.initial_conv(x)
        styles = styles.transpose(0, 1)

        rgb = None
        
        counter = 0
        
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            if counter > 0:
                x, rgb, stage_mask = block(x, rgb, style, input_noise, w, stage_bbox)
            else:
                x, rgb, stage_mask = block(x, rgb, style, input_noise, w, bmask)
                
            
            if counter < self.num_layers - 1:
                hh, ww = x.size(2), x.size(3)
                seman_bbox = batched_index_select(stage_mask, dim = 1, index = y.view(batch_size, num_o, 1, 1))
                seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
                
                alpha = torch.gather(self.sigmoid(getattr(self,'alpha_%s' % (str(counter)))).expand(batch_size, -1, -1), dim=1, index=y.view(batch_size, num_o, 1)).unsqueeze(-1)
                stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha) + seman_bbox * alpha
                
            
                                            
            counter += 1

        return rgb, bmask, stage_bbox
    

def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
    
def bbox_mask(x, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4).cuda()
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)
                               

########################################################end of LostGAN generator###################################################
    
########################################################Lost RCNN discriminator####################################################

class Fp32RoIAlign(RoIAlign):
    """ Workaround for FP16 traning
    """
    def __init__(
            self,
            output_size,
            spatial_scale,
            sampling_ratio,
            aligned=False,
    ):
        super().__init__(output_size, spatial_scale, sampling_ratio, aligned)

    def forward(self, input, rois):
        output = super().forward(input.float(), rois.float())
        return output.type_as(input)
    

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)
    
def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv
    




class RCNN_Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, transparent = False, fmap_max = 512, num_classes = 10, input_dim = 3):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)
            
            ############# lostgan related components #################
            
            if ind == 2:
                self.block_obj3 = ResBlock(in_chan, out_chan, downsample = False)
            elif ind == 3:
                self.block_obj4 = ResBlock(in_chan, out_chan, downsample = False)
            elif ind == 4:
                self.block_obj5 = ResBlock(in_chan, out_chan, downsample = True)
                self.l_obj = nn.utils.spectral_norm(nn.Linear(out_chan, 1))
                self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, out_chan))
                                                    
            ##########################################################
        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)
        
        ############################ LostGAN related settings ############################
        
        self.roi_align_s = Fp32RoIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = Fp32RoIAlign((8, 8), 1.0 / 8.0, int(0))
        self.activation = nn.ReLU()
        
        
        ##################################################################################
        
    def process(self, x, bbox = None, y = None):
        idx = torch.arange(start=0, end=x.size(0),
                           device=x.device).view(x.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox_ = bbox.clone()
        
        bbox_[:, :, 2] = bbox_[:, :, 2] + bbox_[:, :, 0]
        bbox_[:, :, 3] = bbox_[:, :, 3] + bbox_[:, :, 1]
        bbox_ = bbox_ * x.size(2)
        bbox_ = torch.cat((idx, bbox_.cuda().float()), dim=2)
        bbox_ = bbox_.view(-1, 5)
        y = y.view(-1)

        idx = (y != 0).nonzero().view(-1)
        bbox_ = bbox_[idx]
        y = y[idx]
        
        return idx, bbox_, y

    def forward(self, x, bbox = None, y = None, cutmix = False):
        
        # b, *_ = x.shape
        
        idx, bbox, y = self.process(x, bbox, y)
        
        b = bbox.size(0)
        
        residuals = []
        
        counter = 0
        
        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)
                
            if counter == 1:
                x1 = x
            elif counter == 2:
                x2 = x
            
            counter += 1
            
            
        x = self.conv(x) + x
        enc_out = self.to_logit(x)
        
        ############## lostgan related components #########################
        
        # obj path
        # seperate different path
        
        
        
        if cutmix == False:
            s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
            bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
            y_l, y_s = y[~s_idx], y[s_idx]
        
            obj_feat_s = self.block_obj3(x1)
            obj_feat_s = self.block_obj4(obj_feat_s)
            obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)
            
            
        
            obj_feat_l = self.block_obj4(x2)
            obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)
            
            
        
            obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim = 0)
            
            y = torch.cat([y_l, y_s], dim = 0)
        
            obj_feat = self.block_obj5(obj_feat)
            obj_feat = self.activation(obj_feat)
            obj_feat = torch.sum(obj_feat, dim = (2, 3))
            out_obj = self.l_obj(obj_feat)
            
            # print(out_obj.size())
            
            out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim = 1, keepdim = True)
        
        elif cutmix == True:
            out_obj = None
        
        ###################################################################

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out, out_obj


######################################################## end of Lost RCNN discriminator ############################################ 
    
    
class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, steps = 1, lr = 1e-4, ttur_mult = 2, no_const = False, lr_mul = 0.1, aug_types = ['translation', 'cutout']):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const)

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size, aug_types)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()

        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1')

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
    
############################################### Lost StyleGAN v2 ############################################################


class Lost_StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, steps = 1, lr = 1e-4, ttur_mult = 2, no_const = False, lr_mul = 0.1, aug_types = [], num_classes = 10, z_dim = 128):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.G = Lost_Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const, fmap_max = fmap_max, num_classes = num_classes, z_dim = z_dim)
        
        self.D = RCNN_Discriminator(image_size, network_capacity, transparent = transparent, fmap_max = fmap_max, num_classes = num_classes)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.GE = Lost_Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const, num_classes = num_classes)

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size, aug_types)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()

        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1')

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x



############################################### End of Lost StyleGAN v2 ####################################################


class Trainer():
    def __init__(self, name, results_dir, models_dir, image_size, network_capacity, out_path, transparent = False, batch_size = 4, mixed_prob = 0.9, gradient_accumulate_every=1, lr = 2e-4, ttur_mult = 2, num_workers = None, save_every = 1000, trunc_psi = 0.6, fp16 = False, no_const = False, aug_prob = 0., dataset_aug_prob = 0., cr_weight = 0.2, apply_pl_reg = False, lr_mul = 0.1, gan_type = None,  dataset = 'coco', z_dim = 128, opt = None, *args, **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None
        
        self.out_path = out_path
        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'
        self.batch_size = batch_size

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent

        self.no_const = no_const
        self.aug_prob = aug_prob

        self.lr = lr
        self.ttur_mult = ttur_mult
        self.lr_mul = lr_mul
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0
        
        self.epochs = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.apply_pl_reg = apply_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.cr_weight = cr_weight
        self.gan_type = gan_type
        self.dataset = dataset
        
        vgg_loss = VGGLoss()
        vgg_loss = nn.DataParallel(vgg_loss)
        self.vgg_loss = vgg_loss
        
        self.l1_loss = nn.DataParallel(nn.L1Loss())
        
        self.z_dim = z_dim
        if self.dataset == "coco":
            self.num_obj = 8
            
        self.lamb_obj = 1.0
        self.lamb_img = 1.0
        
        self.opt = opt

    def init_GAN(self):
        args, kwargs = self.GAN_params
        
        if self.dataset == 'coco':
            num_classes = 184
            num_ojb = 8
        elif self.dataset == 'mnist':
            num_classes = 10
            num_obj = 9
        
        z_dim = 128
        
        if self.gan_type != 'lost':
            self.GAN = StyleGAN2(lr = self.lr, ttur_mult = self.ttur_mult, lr_mul = self.lr_mul, image_size = self.image_size, network_capacity = self.network_capacity, transparent = self.transparent, fp16 = self.fp16, no_const = self.no_const, *args, **kwargs)
        elif self.gan_type == 'lost':
            self.GAN = Lost_StyleGAN2(lr = self.lr, ttur_mult = self.ttur_mult, lr_mul = self.lr_mul, image_size = self.image_size, network_capacity = self.network_capacity, transparent = self.transparent, fp16 = self.fp16, no_const = self.no_const, num_classes = num_classes, z_dim = z_dim, *args, **kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.no_const = config.pop('no_const', False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        self.loader = cycle(data.DataLoader(self.dataset, num_workers = default(self.num_workers, num_cores), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))
        
    def get_dataset(self, dataset, img_size):
        
        if dataset == "coco":
            data = CocoSceneGraphDataset(image_dir='/home/jshen27/LOSTGAN_exp_two_level_part/datasets/coco/images/train2017/',
                                        instances_json='/home/jshen27/LOSTGAN_exp_two_level_part/datasets/coco/annotations/instances_train2017.json',
                                        stuff_json='/home/jshen27/LOSTGAN_exp_two_level_part/datasets/coco/annotations/stuff_train2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=True)
        elif dataset == 'mnist':
            transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        # data = MNISTDataset(root=mnist_dir,
                            # transform=transform)
            data = MNISTDataset(transform=transform)
        
        elif dataset == 'vg':
            data = VgSceneGraphDataset(vocab=vocab, h5_path='./datasets/vg/train.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(img_size, img_size), max_objects=10, left_right_flip=True)
        
        return data
            
    
########################################## train one epoch ############################################################    
    def train_epoch(self):
        
        if (self.epochs + 1) % self.opt.epoch_interval == 0:
            self.opt.outf = '%s/train_epoch_%d' % (self.opt.out_, self.epochs)
            try:
                os.makedirs(self.opt.outf)
            except OSError:
                pass
        
        
        if self.loader == None:
            train_data = self.get_dataset(self.dataset, self.image_size)
            self.loader = torch.utils.data.DataLoader(
                train_data, batch_size = self.batch_size,
                drop_last = True, shuffle = True, num_workers = 0)
            if self.dataset == 'mnist':
                self.idx_to_name_dic = None
                self.num_classes = 10
                self.num_obj = 9
            elif self.dataset == 'coco':
                self.idx_to_name_dic = train_data.vocab['object_idx_to_name']
                self.num_classes = 184
                self.num_obj = 8
        
        if self.GAN is None:
            self.init_GAN()
            
            self.GAN.G = DataParallelWithCallback(self.GAN.G)
            # self.GAN.G = nn.DataParallel(self.GAN.G)
            self.GAN.D = nn.DataParallel(self.GAN.D)
            
            
            
            
        self.GAN.train()
        
        image_size = self.opt.image_size
        latent_dim = self.opt.latent_dim
        num_layers = int(log2(image_size) - 1)
        # num_layers = self.GAN.G.num_layers
        
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        
        z_dim = self.opt.z_dim

        aug_prob   = self.aug_prob
        
        dec_loss_coef = warmup(0, 1, 20, self.epochs)
        cutmix_prob = warmup(0, 0.25, 20, self.epochs)
        
        
        
        backwards = partial(loss_backwards, self.fp16)
        
        # print(self.GAN.G.module.alpha_1)
        
        
        for idx, data in enumerate(self.loader):
            
            print('iteration')
            print(idx)
            
            total_disc_loss = torch.tensor(0.).cuda()
            total_gen_loss = torch.tensor(0.).cuda()
            
            apply_path_penalty = self.apply_pl_reg and idx % 32 == 0
            
            if self.epochs == 0:
                if idx < 4000:
                    apply_gradient_penalty = True
                else:
                    apply_gradient_penalty = idx % 4 == 0
            else:
                apply_gradient_penalty = idx % 4 == 0
            
            
            
            apply_cutmix = random.random() < cutmix_prob
            
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float()
            real_images.requires_grad_()
            
            
            
            avg_pl_length = self.pl_mean
            self.GAN.D_opt.zero_grad()
            
            # train discriminator
            
            for i in range(self.gradient_accumulate_every):
                get_latents_fn = mixed_list if random.random() < self.mixed_prob else noise_list
                style = get_latents_fn(batch_size, num_layers, latent_dim)
                noise = image_noise(batch_size, image_size)
                
                w_space = latent_to_w(self.GAN.S, style)
                w_styles = styles_def_to_tensor(w_space)
                
                z_obj_style = torch.randn(real_images.size(0), self.num_obj, z_dim).cuda()
                
                
                
                generated_images, bmask, stage_bbox = self.GAN.G(w_styles, noise, z_obj_style, bbox, y = label.squeeze(dim = -1))
                
               
                
                (fake_enc_out, fake_dec_out, d_out_fobj), fake_aug_images = self.GAN.D_aug(generated_images.detach(), bbox, label, detach = True, prob = aug_prob)
                
                  
                      
                
                
                (real_enc_out, real_dec_out, d_out_robj), real_aug_images = self.GAN.D_aug(real_images, bbox, label, prob = aug_prob)
                
                
                      
                      
                enc_divergence = (F.relu(1 + real_enc_out) + F.relu(1 - fake_enc_out)).mean()
                dec_divergence = (F.relu(1 + real_dec_out) + F.relu(1 - fake_dec_out)).mean()
                img_loss = enc_divergence + dec_divergence * dec_loss_coef
                
                d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
                d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
                obj_loss = d_loss_robj + d_loss_fobj
                
                divergence = self.lamb_obj * obj_loss + self.lamb_img * img_loss
                
                disc_loss = divergence
                
                if apply_cutmix:
                    mask = cutmix(
                    torch.ones_like(real_dec_out),
                    torch.zeros_like(real_dec_out),
                    cutmix_coordinates(image_size, image_size)
                    )
                    
                    if random.random() > 0.5:
                        mask = 1 - mask

                    cutmix_images = mask_src_tgt(real_aug_images, fake_aug_images, mask)
                    cutmix_enc_out, cutmix_dec_out, d_obj_None = self.GAN.D(cutmix_images, bbox, label, cutmix = True)
                    
                    cutmix_enc_divergence = F.relu(1 - cutmix_enc_out).mean()
                    cutmix_dec_divergence =  F.relu(1 + (mask * 2 - 1) * cutmix_dec_out).mean()
                    disc_loss = disc_loss + cutmix_enc_divergence + cutmix_dec_divergence
                    
                    cr_cutmix_dec_out = mask_src_tgt(real_dec_out, fake_dec_out, mask)
                    cr_loss = F.mse_loss(cutmix_dec_out, cr_cutmix_dec_out) * self.cr_weight
                    self.last_cr_loss = cr_loss.clone().detach().item()
                    
                    disc_loss = disc_loss + cr_loss * dec_loss_coef
                    
               
                 
                if apply_gradient_penalty:
                    if random.random() < 0.5:
                        gp = gradient_penalty(real_images, (real_enc_out,))
                    else:
                        gp = gradient_penalty(real_images, (real_dec_out,)) * dec_loss_coef
                    self.last_gp_loss = gp.clone().detach().item()
                    disc_loss = disc_loss + gp
                
                
                disc_loss = disc_loss / self.gradient_accumulate_every
                disc_loss.register_hook(raise_if_nan)
                backwards(disc_loss, self.GAN.D_opt)

                total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every
             
            self.d_loss = float(total_disc_loss)
            self.GAN.D_opt.step()
            
            
            # train generator
            
            self.GAN.G_opt.zero_grad()
            
            for i in range(self.gradient_accumulate_every):
                
                (fake_enc_output, fake_dec_output, d_out_fobj), fake_aug_images = self.GAN.D_aug(generated_images, bbox, label, prob = aug_prob)
                
                g_loss_fake = fake_enc_output.mean() + F.relu(1 + fake_dec_output).mean()
                g_loss_obj = -d_out_fobj.mean()
                
                pixel_loss = self.l1_loss(generated_images, real_images).mean()
                feat_loss = self.vgg_loss(generated_images, real_images).mean()
                
                gen_loss = self.lamb_obj * g_loss_obj + self.lamb_img * g_loss_fake + pixel_loss + feat_loss
                
                if apply_path_penalty:
                    pl_lengths = calc_pl_lengths(w_styles, generated_images)
                    avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                    if not is_empty(self.pl_mean):
                        pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                        if not torch.isnan(pl_loss):
                            gen_loss = gen_loss + pl_loss
                
                gen_loss = gen_loss / self.gradient_accumulate_every
                gen_loss.register_hook(raise_if_nan)
                backwards(gen_loss, self.GAN.G_opt)
                
                total_gen_loss += g_loss_fake.detach().item() / self.gradient_accumulate_every
            
            self.g_loss = float(total_gen_loss)
            self.GAN.G_opt.step()
            
            # calculate moving averages
            
            if apply_path_penalty and not np.isnan(avg_pl_length):
                self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

            # if self.steps % 10 == 0 and self.steps > 20000:
                # self.GAN.EMA()

            # if self.steps <= 25000 and self.steps % 1000 == 2:
                # self.GAN.reset_parameter_averaging()
                
            # save from NaN errors
            
            # checkpoint_num = floor(self.steps / self.save_every)

            # if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
                # print(f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}')
                # self.load(checkpoint_num)
                # raise NanException

            # periodically save results

            # if self.steps % self.save_every == 0:
                # self.save(checkpoint_num)

            # if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                # self.evaluate(floor(self.steps / 1000))
            
            # for sanity check
            # if (idx + 1) % self.opt.idx_interval == 0:
                # break
        
        if (self.epochs + 1) % self.opt.epoch_interval == 0:
            torch.save(self.GAN.state_dict(), '%s/GAN.pth' % (self.opt.outf))
            
            self.plot_results(real_images[0:5], label[0:5], bbox[0:5], self.idx_to_name_dic, self.opt)
        
        self.epochs += 1
        self.av = None
    
    def plot_results(self, real_images, label, bbox, idx_to_name_dic, opt):
        
        
        
        img_size = self.GAN.G.module.image_size
        latent_dim = self.GAN.G.module.latent_dim
        num_layers = self.GAN.G.module.num_layers
        z_dim = 128
        
        
        if opt.dataset == 'coco':
            num_classes = 184
            num_obj = 8
        elif opt.dataset == 'mnist':
            num_classes = 10
            num_obj = 9
        elif opt.dataset == 'vg':
            num_classes = 179
            num_obj = 31
            
        
        b,_,_,_ = real_images.size()
        
        with torch.no_grad():
            samples = []
            for trials in range(5):
                get_latents_fn = mixed_list if random.random() < self.mixed_prob else noise_list
                style = get_latents_fn(b, num_layers, latent_dim)
                noise = image_noise(b, img_size)
               
                w_space = latent_to_w(self.GAN.S, style)
                w_styles = styles_def_to_tensor(w_space)
                
                z_obj_style = torch.randn(b, num_obj, z_dim).cuda()
                
                fake_images, bmask, stage_bbox = self.GAN.G(w_styles, noise, z_obj_style, bbox, y = label.squeeze(dim = -1))
                
                samples.append(fake_images)
        
        fig, ax = plt.subplots(5, 7)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.1, hspace = 0)
        
        A = torch.zeros((128, 128), dtype = int)
        
        for row in range(5):
            obj_labels = label[row]
            obj_bbox = bbox[row]
            real = real_images[row]
            
            for col in range(7):
                if col == 0:
                    ax[row, col].imshow(A, cmap = 'binary', vmin = 0, vmax = 1)
                    ax[row, col].set_yticks([])
                    ax[row, col].set_xticks([])
                    
                    for k in range(num_obj):
                        label_index = obj_labels[k]
                        if opt.dataset == 'mnist' or label_index != 0:
                            if opt.dataset == 'mnist':
                                label_name = str(label_index.item())
                            else:
                                label_name = idx_to_name_dic[label_index]
                            
                            x, y, w, h = obj_bbox[k]
                            
                            X = int(128 * x)
                            Y = int(128 * y)
                            W = int(128 * w)
                            H = int(128 * h)
                            
                            ax[row, col].add_patch(Rectangle((X, Y), W, H, fill = None, alpha = 1))
                            centerx = X + 0.5 * W
                            centery = Y + 0.5 * H
                            ax[row, col].text(centerx, centery, label_name)
                            
                       
                
                elif col == 1:
                    ax[row, col].imshow((real.cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                    ax[row, col].set_title('real_image', fontsize = 10)
                    ax[row, col].set_yticks([])
                    ax[row, col].set_xticks([])
                
                else:
                    ax[row, col].imshow((samples[col - 2][row].cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                    ax[row, col].set_title('generated_image', fontsize = 10)
                    ax[row, col].set_yticks([])
                    ax[row, col].set_xticks([])
                
         
        fig.savefig('%s/fix_layout.png' % (opt.outf))
        
        selected_label_entries, l = self.get_wanted_labels(label, opt)
        
        draw_masks = []
        
        if len(l) != 0:
            for i in range(len(selected_label_entries)):
                sample = selected_label_entries[i]
                if sample != []:
                    for j in range(len(sample)):
                        position = sample[j]
                        label_index = label[i, position]
                        if opt.dataset == 'mnist':
                            label_name = str(label_index.item())
                        
                        else:
                            label_name = idx_to_name_dic[label_index]
                            
                        temp_l = [label_name, bmask[i, position, :, :], stage_bbox[i, position, :, :], bbox[i, position], real_images[i], fake_images[i]]
                        draw_masks.append(temp_l)
                        
        
          
            for k in range(len(l)):
                sample = draw_masks[k]
                obj_name = sample[0]
                init_mask = sample[1]
                refine_mask = sample[2]
                x, y, w, h = sample[3]
                
               
                
                real_image = sample[4]
                fake_image = sample[5]
                
                fig, ax = plt.subplots(1, 6)
                
                fig.set_figheight(15)
                fig.set_figwidth(15)
                fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.1, hspace = 0)
                
                X_ = int(128 * x)
                Y_ = int(128 * y)
                W_ = int(128 * w)
                H_ = int(128 * h)
                
                
                
                
                
                cropped_real = real_image[:, Y_:Y_ + H_,X_: X_ + W_]
                
                
                
                cropped_real = cropped_real.view(1, 3, H_, W_)
                resize_cropped_real = F.interpolate(cropped_real, size = (64, 64), mode = 'nearest')
                
                cropped_fake = fake_image[:, Y_:Y_ + H_,X_: X_ + W_]
                cropped_fake = cropped_fake.view(1, 3, H_, W_)
                resize_cropped_fake = F.interpolate(cropped_fake, size = (64, 64), mode = 'nearest')
                
                X = int(64 * x)
                Y = int(64 * y)
                W = int(64 * w)
                H = int(64 * h)
                
                for col in range(6):
                    if col == 0:
                        ax[col].imshow((real_image.cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                        ax[col].add_patch(Rectangle((X_, Y_), W_, H_, fill = None, alpha = 1, color = 'red'))
                        ax[col].set_title('real_image', fontsize = 10)
                    elif col == 1:
                        ax[col].imshow((resize_cropped_real[0,:,:,:].cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                        ax[col].set_title('real' + obj_name)
                    elif col == 2:
                        ax[col].imshow((fake_image.cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                        ax[col].add_patch(Rectangle((X_, Y_), W_, H_, fill = None, alpha = 1, color = 'red'))
                        ax[col].set_title('fake_image', fontsize = 10)
                    elif col == 3:
                        ax[col].imshow((resize_cropped_fake[0,:,:,:].cpu().data * 0.5 + 0.5).permute(1, 2, 0).numpy())
                        ax[col].set_title('fake' + obj_name)
                    elif col == 4:
                        cropped_init = init_mask[Y:Y + H, X:X + W]
                        cropped_init = cropped_init.view(1, 1, H, W)
                        resize_cropped_init = F.interpolate(cropped_init, size = (64, 64), mode = 'nearest')
                        ax[col].imshow(resize_cropped_init[0, 0, :, :].cpu().detach().numpy())
                        ax[col].set_title(obj_name + '_init_mask', fontsize = 10)
                    elif col == 5:
                        cropped_refine = refine_mask[Y:Y + H, X:X + W]
                        cropped_refine = cropped_refine.view(1, 1, H, W)
                        resize_cropped_refine = F.interpolate(cropped_refine, size = (64, 64), mode = 'nearest')
                        ax[col].imshow(resize_cropped_refine[0, 0, :, :].cpu().detach().numpy())
                        ax[col].set_title(obj_name + '_refine_mask', fontsize = 10)
                    
                    ax[col].set_yticks([])
                    ax[col].set_xticks([])
                    
                fig.savefig('%s/predict_mask_%s.png' % (opt.outf, str(k)))    
                    
                    
                    
                           
    
    
    
    def get_wanted_labels(self, label, opt):
        
        b,num_obj,_ = label.size()
        selected_label_entries = []
        l = []
        
        for i in range(5):
            
            selected_label_entries.append([])
            obj_labels = label[i]
            
            for j in range(num_obj):
                
                obj_label = obj_labels[j]
                l.append(j)
                selected_label_entries[i].append(j)
                
                
                if len(l) == 10:
                    
                    return selected_label_entries, l
         
        return selected_label_entries, l
            
            
        
                                
            
        
    
    def train(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob

        apply_gradient_penalty = self.steps < 4000 or self.steps % 4 == 0
        apply_path_penalty = self.apply_pl_reg and self.steps % 32 == 0

        dec_loss_coef = warmup(0, 1., 30000, self.steps)
        cutmix_prob = warmup(0, 0.25, 30000, self.steps)
        apply_cutmix = random.random() < cutmix_prob

        backwards = partial(loss_backwards, self.fp16)

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random.random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise).clone().detach()
            
            
            
            
            (fake_enc_out, fake_dec_out), fake_aug_images = self.GAN.D_aug(generated_images, detach = True, prob = aug_prob)
            
            
            

            real_images = next(self.loader).cuda()
            real_images.requires_grad_()
            
            
            
            (real_enc_out, real_dec_out), real_aug_images = self.GAN.D_aug(real_images, prob = aug_prob)

            enc_divergence = (F.relu(1 + real_enc_out) + F.relu(1 - fake_enc_out)).mean()
            dec_divergence = (F.relu(1 + real_dec_out) + F.relu(1 - fake_dec_out)).mean()
            divergence = enc_divergence + dec_divergence * dec_loss_coef

            disc_loss = divergence
            
            

            if apply_cutmix:
                mask = cutmix(
                    torch.ones_like(real_dec_out),
                    torch.zeros_like(real_dec_out),
                    cutmix_coordinates(image_size, image_size)
                )

                if random.random() > 0.5:
                    mask = 1 - mask

                cutmix_images = mask_src_tgt(real_aug_images, fake_aug_images, mask)
                cutmix_enc_out, cutmix_dec_out = self.GAN.D(cutmix_images)

                cutmix_enc_divergence = F.relu(1 - cutmix_enc_out).mean()
                cutmix_dec_divergence =  F.relu(1 + (mask * 2 - 1) * cutmix_dec_out).mean()
                disc_loss = disc_loss + cutmix_enc_divergence + cutmix_dec_divergence

                cr_cutmix_dec_out = mask_src_tgt(real_dec_out, fake_dec_out, mask)
                cr_loss = F.mse_loss(cutmix_dec_out, cr_cutmix_dec_out) * self.cr_weight
                self.last_cr_loss = cr_loss.clone().detach().item()

                disc_loss = disc_loss + cr_loss * dec_loss_coef

            if apply_gradient_penalty:
                if random.random() < 0.5:
                    gp = gradient_penalty(real_images, (real_enc_out,))
                else:
                    gp = gradient_penalty(real_images, (real_dec_out,)) * dec_loss_coef
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            (fake_enc_output, fake_dec_output), _ = self.GAN.D_aug(generated_images, prob = aug_prob)
            loss = fake_enc_output.mean() + F.relu(1 + fake_dec_output).mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        # if self.steps % 10 == 0 and self.steps > 20000:
            # self.GAN.EMA()

        # if self.steps <= 25000 and self.steps % 1000 == 2:
            # self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results

        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, save_frames = False):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        ratios = torch.linspace(0., 8., 100)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        pl_mean = default(self.pl_mean, 0)
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {pl_mean:.2f} | CR: {self.last_cr_loss:.2f}')

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {'GAN': self.GAN.state_dict()}

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        self.GAN.load_state_dict(load_data['GAN'])

        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
