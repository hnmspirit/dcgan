import os, glob
from os.path import basename, isdir, join
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils

from datasets import *
from dcgan64 import *
from utils import *

# random
mySeed = 520
random.seed(mySeed)
torch.manual_seed(mySeed)

# params
dataroot = "celeba"
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64

epochs = 5
lr = 0.0002
beta1 = 0.5

workers = 2
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

iter_print_percent = 0.4
epoch_sample = 5
epoch_checkpoint = 20
epoch_load = 0
path_sample = 'samples'
path_checkpoint = 'models'

# /path/to/celeba
#     -> img_align_celeba
#         -> 188242.jpg

if not isdir(path_sample):
    os.makedirs(path_sample)

if not isdir(path_checkpoint):
    os.makedirs(path_checkpoint)

# data
transform = get_transform(image_size)
dataset = load_celeba(dataroot, transform)
dataloader = load_loader(dataset, batch_size, workers)

# model
netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(ndf, nc).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

# loss
criterion = nn.BCELoss()
label_real = 1
label_fake = 0

# optim
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# sample
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# load model
if epoch_load != 0:
    load_model(netG, join(path_checkpoint, 'netG_{:09d}.pth'.format(epoch_load)))
    load_model(netD, join(path_checkpoint, 'netD_{:09d}.pth'.format(epoch_load)))


# Training Loop
iter_print = int(iter_print_percent * len(dataset) // batch_size)
print('\n' + '-'*5 + "Starting Training Loop..." + '-'*5 + '\n')

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        ## REAL
        img_real = data[0].to(device)
        b_size = img_real.size(0)
        label = torch.full((b_size,), label_real, device=device)
        # Forward pass real batch through D
        output = netD(img_real).view(-1)
        # Calculate loss on all-real batch
        loss_d_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        loss_d_real.backward()
        D_x = output.mean().item()

        ## FAKE
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        img_fake = netG(noise)
        label.fill_(label_fake)
        # Classify all fake batch with D
        output = netD(img_fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        loss_d_fake = criterion(output, label)
        # Calculate the gradients for this batch
        loss_d_fake.backward()
        D_g1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        loss_d = loss_d_real + loss_d_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(label_real)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(img_fake).view(-1)
        # Calculate G's loss based on this output
        loss_g = criterion(output, label)
        # Calculate gradients for G
        D_g2 = output.mean().item()
        # Update G
        loss_g.backward()
        optimizerG.step()

        # Output training stats
        if i % iter_print == 0:
            print('[%d/%d][%4d/%4d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     loss_d.item(), loss_g.item(), D_x, D_g1, D_g2))

    if (epoch % epoch_sample == 0) or (epoch == epochs-1):
        with torch.no_grad():
            img_fake = netG(fixed_noise).detach().cpu()[:128]
        savepath = join(path_sample, 'sample_{:09d}.jpg'.format(epoch))
        vutils.save_image(img_fake, savepath, nrow=8, padding=2, normalize=True)

    if (epoch+1) % epoch_checkpoint:
        save_model(netG, join(path_checkpoint, 'netG_{:09d}.pth'.format(epoch)))
        save_model(netD, join(path_checkpoint, 'netD_{:09d}.pth'.format(epoch)))

