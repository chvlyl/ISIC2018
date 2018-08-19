import argparse
import json
import random
from pathlib import Path
import numpy as np

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorboardX import SummaryWriter

## pytorch
import torch
from torch import nn
from torch.optim import Adam
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import torch.nn.functional as F
#
# ## model
from models import UNet, UNet11, UNet16, LinkNet34
# from loss import LossBinary, LossMulti
# from dataset import make_loader
# from utils import save_weights, write_event, write_tensorboard
# from validation import validation_binary, validation_multi
# from prepare_train_val import get_split
# from validation import get_jaccard
#
# from torchvision import transforms
# from transforms import DualCompose, ImageOnly, Normalize, HorizontalFlip, VerticalFlip
#
# import h5py
# import random
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torchvision.transforms.functional as TF
#
# from keras.preprocessing.image import array_to_img, img_to_array
#
# from dataset import SkinDataset
# ## define data loader
# import importlib
#
# # importlib.reload(dataset)
# # import reload
# # reload(SkinDataset)
#
# image_path = '/media/eric/SSD2/Project/11_ISCB2018/0_Data/Task2/h5/'
# train_test_id = get_split(fold=None)
#
# data_set = SkinDataset(train_test_id=train_test_id, image_path=image_path, train=True,
#                        mask_attr='pigment_network', transform=None,
#                        problem_type='binary')
# data_loader = DataLoader(data_set, batch_size=16,
#                          shuffle=True, num_workers=8,
#                          pin_memory=torch.cuda.is_available())
#
# train_loader = data_loader
# train_image, train_mask = next(iter(train_loader))
# print('train_image.shape', train_image.shape)
# print('train_mask.shape', train_mask.shape)
# print('train_image.min', train_image.min().item())
# print('train_image.max', train_image.max().item())
# print('train_mask.min', train_mask.min().item())
# print('train_mask.max', train_mask.max().item())
#
# num_classes = 4
# model = UNet16(num_classes=num_classes, pretrained='vgg')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# model.to(device)
# state = torch.load('unet16_parts_20/model_0.pt')
# epoch = state['epoch']
# step = state['step']
# model.load_state_dict(state['model'])
# print('Restored model, epoch {}, step {:,}'.format(epoch, step))
#
# model = list(model.children())[0]
# num_filters = 32
# num_classes = 5
# model.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
# model = nn.DataParallel(model)
# model.to(device)
# loss_fn = LossMulti(num_classes=num_classes, jaccard_weight=1)
# criterion = loss_fn
#
# ######
# train_image = train_image.permute(0, 3, 1, 2)
# train_mask = train_mask.permute(0, 3, 1, 2)
# train_image = train_image.to(device)
# train_mask = train_mask.to(device).type(torch.cuda.LongTensor)
# outputs = model(train_image)
# train_prob = outputs
# loss = criterion(outputs, train_mask)
#
# import torch
# import torchvision
#
# train_image = train_image.permute(0, 3, 1, 2)
# train_mask = train_mask.permute(0, 3, 1, 2)
#
# saved_images = torchvision.utils.make_grid(train_image, nrow=train_image.shape[0])
# plt.imshow(saved_images.permute(1, 2, 0))
# plt.show()


# import torch
# from torch import nn
#
# # 2D loss example (used, for example, with image inputs)
# N, C = 5, 4
# loss = nn.NLLLoss()
# # input is of size N x C x height x width
# data = torch.randn(N, 16, 10, 10)
# m = nn.Conv2d(16, C, (3, 3))
# # each element in target has to have 0 <= value < C
# target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
# output = m(data)
# loss_value = loss(output, target)
# #output.backward()

#
# num_classes = 10
# model = UNet16(num_classes=num_classes, pretrained='vgg')
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# model.to(device)
#
# model_weight = 'weights/TernausNet_UNet16_Skin_Pretrain.pt'
#
# state = torch.load(model_weight)
# epoch = state['epoch']
# step = state['step']
# model.load_state_dict(state['model'])
#
# model = list(model.children())[0]
# num_filters = 32
# model.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
# print('--' * 10)
# print('Load pretrained model and replace the last layer', model_weight, num_classes)
# print('--' * 10)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# model.to(device)
#
# # for name, para in model.named_parameters():
# #     print(name, para.numel(), para.requires_grad)
# #     if name == 'module.final.weight':
# #         para.requires_grad = False
#
#
# def freeze_layers(model, freeze_layer_names=None, train_layer_names=None):
#     for name, para in model.named_parameters():
#         #print(name, para.numel(), para.requires_grad)
#         if name not in train_layer_names:
#             print('Freeze ->', name)
#             para.requires_grad = False
#
# freeze_layers(model, freeze_layer_names=None, train_layer_names=['module.final.weight','module.final.bias'])
#
# for name, para in model.named_parameters():
#     print(name, para.numel(), para.requires_grad)
#

#model.final.weight.requires_grad = False
#model.final.bias.requires_grad = False
#
# import pickle
# with open('/media/eric/SSD2/Project/11_ISCB2018/2_Analysis/20180516_classification_segmentation/train_test_id.pickle',
#           'rb') as f:
#     mask_ind = pickle.load(f)
#
# mask_ind['total'] = mask_ind[['pigment_network',
#                               'negative_network',
#                               'streaks',
#                               'milia_like_cyst',
#                               'globules']].sum(axis=1)
#
# train_test_id = mask_ind
# train_test_id = train_test_id[~((train_test_id.Split == 'train') & (train_test_id.total <= 2))]
#
#
# def binary_cross_entropy_with_logits_for_multiple_masks(input, target, weight=None):
#     if not (target.size() == input.size()):
#         raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
#
#     max_val = (-input).clamp(min=0)
#     loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#
#     loss.mean()
#
#     if weight is not None:
#         loss = loss * weight


from torchnet.meter import AUCMeter
#
# import torch
# #temp = torch.cat((torch.ones(2,5,3,3), torch.ones(2,5,3,3)*2), 1)
# torch.ones(2,5,3,3)*
#
# mtr = meter.AUCMeter()
#
# test_size = 1000
# mtr.add(torch.rand(test_size), torch.zeros(test_size))
# mtr.add(torch.rand(test_size), torch.Tensor(test_size).fill_(1))
#
# val, tpr, fpr = mtr.value()
# self.assertTrue(math.fabs(val - 0.5) < 0.1, msg="AUC Meter fails")
#
# mtr.reset()
# mtr.add(torch.Tensor(test_size).fill_(0), torch.zeros(test_size))
# mtr.add(torch.Tensor(test_size).fill_(0.1), torch.zeros(test_size))
# mtr.add(torch.Tensor(test_size).fill_(0.2), torch.zeros(test_size))
# mtr.add(torch.Tensor(test_size).fill_(0.3), torch.zeros(test_size))
# mtr.add(torch.Tensor(test_size).fill_(0.4), torch.zeros(test_size))
# mtr.add(torch.Tensor(test_size).fill_(1),
#         torch.Tensor(test_size).fill_(1))
# val, tpr, fpr = mtr.value()
#
#
# import torchvision
# #'densenet121', 'densenet169', 'densenet201', 'densenet161'
# model = torchvision.models.vgg16(pretrained=True)
# for ind,(name, para) in enumerate(model.named_parameters()):
#     print(ind, name, para.numel(), para.requires_grad)
#
#
# encoder = torchvision.models.vgg16(pretrained=True).features
# print(model)

import re
from models import UNet, UNet11, UNet16, UNet16BN, LinkNet34
model = UNet16BN(num_classes=5, pretrained='vgg')
for ind,(name, para) in enumerate(model.named_parameters()):
    #if re.search('encoder',name):
    print(ind, name, para.numel(), para.requires_grad)