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

def generate_batch_images(_g, batch_size, start=0, end=4, prefix="", suffix="", figure_path="./samples"):
    if start >= end:
        raise ArithmeticError("start is higher than end [%d > %d]" % (start, end))
    pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)    

    for n in range(start, end):
        label = np.full((batch_size, 1), n)
        label_one_hot = (np.arange(5) == label[:,None]).astype(np.float)
        label_one_hot = torch.from_numpy(label_one_hot)
        noise = torch.FloatTensor(batch_size, 48, 5, 5).normal_()
        im_outputs = _g(label_one_hot.float(), noise)
        for i, img in enumerate(im_outputs):
            a = img
                
            img = transforms.ToPILImage()(a)
            #img = img.convert('RGB')
            img.save(os.path.join(figure_path, "%s-%d-%d-%s.png" % (prefix, n, i, suffix)))
