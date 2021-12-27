import os.path as osp
import glob
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import  torch.optim as topt
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision
from torchvision import models, transforms
from tqdm import tqdm
import datetime


torch.manual_seed(10000)
np.random.seed(10000)
random.seed(10000)

save_path = "../my_weight.pt"
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_epochs = 50
batch_size = 8

def make_net(list_class):

    # load vgg 16 pretrained
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=len(list_class))
    # loss
    criterior = nn.CrossEntropyLoss()
    # optimizer
    update_params_name = ["classifier.6.weight", "classifier.6.bias"]
    params_to_update = []

    for name, param in net.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    optimizer = topt.SGD(params=params_to_update, lr=0.001, momentum=0.8)

    return {"net": net, "criterior": criterior, "optimizer": optimizer}

