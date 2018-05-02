import dataset
import os

import sys
import argparse
import time
import pathlib

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

from dataset import ImageFilelist
from utils import generate_batch_images
import model
from pytorch_ssim import SSIM
from pytorch_msssim import MSSSIM

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", help="Number of epochs to run", type=int, default=100)
    parser.add_argument("--sample_output", help="Output of in-training samples", default="./training")
    parser.add_argument("--sample_nums", help="Number of times to produce in-training samples", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--gpu", help="Which GPU to use", type=int, default=0)
    parser.add_argument("--network", help="Pretrained model name", default="vgg16")
    parser.add_argument("--view_image", help="Interactively view images", action="store_true")
    parser.add_argument("--sample_interval", help="Number of epochs between samples", type=int, default=10)
    parser.add_argument("--dataset", help="Caffe list dir", default="/mnt/datadisk/DeepLearning/xc-list-labelled")

    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    args = parse_args()

    #Params
    epochs = args.epoch
    sample_output = args.sample_output
    sample_nums = args.sample_nums
    batch_size = args.batch_size
    gpu_n = args.gpu
    sample_interval = args.sample_interval

    torch.cuda.set_device(gpu_n)

    learning_rate = 1e-3

    dataset = ImageFilelist("./data", "../Textures/texture_list_sample_50", transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor()
    ]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    D = model.Discriminator(6)
    G = model.Generator(5)

    D.apply(weights_init)
    G.apply(weights_init)

    D.cuda(0)
    G.cuda(0)
    print(D)
    print(G)
    D_criterion = torch.nn.BCEWithLogitsLoss().cuda(0)
    D_optimizer = torch.optim.SGD(D.parameters(), lr=1e-3)

    G_criterion = torch.nn.BCEWithLogitsLoss().cuda(0)
    G_l1 = torch.nn.L1Loss().cuda(0)
    G_msssim = MSSSIM().cuda(0)
    G_ssim = SSIM().cuda(0)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-3)

    pathlib.Path(sample_output).mkdir(parents=True, exist_ok=True)

    d_loss = 0
    g_loss = 0
    
    d_to_g_threshold = 0.5
    g_to_d_threshold = 0.3

    train_d = True
    train_g = True

    conditional_training = True
    for epoch in tqdm(range(epochs)):
        for i, (image, label) in enumerate(data_loader):
            if conditional_training:
                if d_loss - g_loss > d_to_g_threshold:
                    train_d = True
                    train_g = False
                elif g_loss - d_loss > g_to_d_threshold:
                    train_g = True
                    train_d = False
                else:
                    train_d = True
                    train_g = True

            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            if train_d:
                real_output = D(Variable(image.cuda(0)))
                one_hot_label = torch.from_numpy((np.arange(6) == (label.numpy()[:, None])).astype(float)).float().cuda(0)
                real_loss = D_criterion(real_output, Variable(one_hot_label))
                real_loss.backward()
                D_optimizer.step()

                fake_label = np.array([5 for i in range(len(image))])
                G_fake_label = np.random.randint(0, 5, len(image))
                
                fake_label_input = (np.arange(5) == (G_fake_label[:,None])).astype(float)
                fake_label_output = (np.arange(6) == (fake_label[:,None])).astype(float)
                noise = torch.FloatTensor(len(image), 48, 5, 5).normal_().cuda(0)

                fake_images = G(Variable(torch.from_numpy(fake_label_input).float().cuda(0)), Variable(noise))
                fake_output = D(fake_images.detach())

                fake_loss = D_criterion(fake_output, Variable(torch.from_numpy(fake_label_output).float().cuda(0)))
                fake_loss.backward()
                D_optimizer.step()

                total_loss = real_loss + fake_loss
                d_loss = total_loss.data[0]

            if train_g:
                fake_label = np.random.randint(0, 5, len(image))
                fake_label_input = (np.arange(5) == (fake_label[:,None])).astype(float)
                fake_noise = torch.FloatTensor(len(image), 48, 5, 5).normal_().cuda(0)

                generated_images = G(Variable(torch.from_numpy(fake_label_input).float().cuda(0)), Variable(fake_noise))
                generated_output = D(generated_images.detach())

                target_label = (np.arange(6) == (fake_label[:,None])).astype(float)
                generator_loss = G_criterion(generated_output, Variable(torch.from_numpy(target_label).float().cuda(0)))
                g_loss = generator_loss.data[0]
                generator_loss.backward()
                G_optimizer.step()

                # Spatial AE
                input_label = (np.arange(5) == (label.numpy()[:,None])).astype(float)
                input_label = torch.from_numpy(input_label).cuda(0)
                fake_noise = torch.FloatTensor(len(image), 48, 5, 5).normal_().cuda(0)

                ae_images = G(Variable(input_label.float()), Variable(fake_noise))
                #l1_loss = G_l1(ae_images, Variable(image.cuda(0)))
                ae_loss = G_msssim(ae_images, Variable(image.cuda(0)))
                
                if np.isnan(torch.mean(ae_loss.data)):
                    ae_loss = G_ssim(ae_images, Variable(image.cuda(0)))

                #ae_loss = l1_loss - ae_loss
                ae_loss.backward()
                G_optimizer.step()

            print("Epoch [%d/%d], Iter [%d/%d] D Loss:%.8f, G Loss: %.8f, AE Loss: %.8f" % (epoch + 1, epochs,
                                                                i, len(dataset) // batch_size, torch.mean(total_loss.data), torch.mean(generator_loss.data), -torch.mean(ae_loss.data)), end="\r")
        print("Epoch [%d/%d], Iter [%d/%d] D Loss:%.8f, G Loss: %.8f, AE Loss: %.8f" % (epoch + 1, epochs,
                                                                i, len(dataset) // batch_size, torch.mean(total_loss.data), torch.mean(generator_loss.data), -torch.mean(ae_loss.data)), end="\n")
        generate_batch_images(G, 5, figure_path=sample_output, prefix="epoch-%d" % (epoch+1))
