# -*- coding: utf-8 -*-
import os
import torch
import torchvision as tv

dataset=tv.datasets.ImageFolder('data/')
for root,dirs,files in os.walk('data/'):
    print(root)
    print(dirs)
    print(files)