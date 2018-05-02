import dataset
import os

import sys
import argparse
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from torch.autograd import Variable

import numpy as np

import model
from dataset import ImageFilelist

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", help="Number of epochs to run", type=int, default=100)
    parser.add_argument("--sample_output", help="Output of in-training samples", default="./training")
    parser.add_argument("--sample_nums", help="Number of times to produce in-training samples", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--gpu", help="Which GPU to use", type=int, default=0)
    parser.add_argument("--network", help="Pretrained model name", default="vgg16")
    parser.add_argument("--view_image", help="Interactively view images", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = parse_args()

    size = (128, 128)

    model = model.Discriminator(5)
    model

    print(model.parameters)

    noise = torch.FloatTensor(args.batch_size, 3, size[0], size[1]).normal_()

    output = model(noise)
