import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np

import PIL

def generate_batch_images(_g, batch_size, start=0, end=5, prefix="", suffix="", figure_path="./samples"):
    if start >= end:
        raise ArithmeticError("start is higher than end [%d > %d]" % (start, end))
    pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)    
    _g.cuda(0)
    for n in range(start, end):
        label = np.full((batch_size, 1), n)
        label_one_hot = (np.arange(5) == label[:,None]).astype(np.float)
        label_one_hot = torch.from_numpy(label_one_hot)
        noise = torch.FloatTensor(batch_size, 48, 5, 5).normal_().cuda(0)
        im_outputs = _g(Variable(label_one_hot.float().cuda(0)), Variable(noise))
        for i, img in enumerate(im_outputs):
            a = img.data.cpu()
                
            img = transforms.ToPILImage()(a)
            img = img.convert('RGB')
            img.save(os.path.join(figure_path, "%s-%d-%d-%s.jpg" % (prefix, n, i, suffix)))
