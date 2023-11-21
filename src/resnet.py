# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo

import pickle
import numpy as np
import cv2
import os

from db import database

# configs for histogram
RES_model  = 'resnet152'  # model type
pick_layer = 'avg'        # extract feature of this layer
d_type     = 'd1'         # distance type

depth = 3  # retrieved depth, set to None will count the ap for whole database

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResidualNet(ResNet):
  def __init__(self, model=RES_model, pretrained=True):
    if model == "resnet18":
        super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    elif model == "resnet34":
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    elif model == "resnet50":
        super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    elif model == "resnet101":
        super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    elif model == "resnet152":
        super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
    max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
    Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
    Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
    avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    fc = self.fc(avg)  # fc.size = N * 1000
    output = {
        'max': Max,
        'avg': avg,
        'fc' : fc
    }
    return output

class ResNetFeat(object):

  def make_samples(self, data, category, verbose=True):
    sample_cache = '{}-{}'.format(RES_model, pick_layer)
    
    temp = os.path.join(os.getcwd(), 'cache')
    cache_dir = category + ' cache'

    data_img = data['img'].tolist()
    db_img = data['img'][1:].tolist()

    try:
      samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
      samples_img = [sample['img'] for sample in samples]

    except Exception as e: #캐시에 아무 것도 없는 경우
      print(e)
      samples = []
      samples_img = []
    
    fin_sample = []
    for img in data_img:
      if img in samples_img:
        sample = next((sample for sample in samples if sample['img'] == img), None)
        sample['hist'] /= np.sum(sample['hist'])  # normalize
        fin_sample.append(sample)
      else:
        for item in data.itertuples():
          if item.img == img:
            d = item
            break
 
        res_model = ResidualNet(model=RES_model)
        res_model.eval()

        if use_gpu:
          res_model = res_model.cuda()

        d_img, d_category = getattr(d, "img"), getattr(d, "category")
        img = cv2.imread(d_img)
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= means[0]  # reduce B's mean
        img[1] -= means[1]  # reduce G's mean
        img[2] -= means[2]  # reduce R's mean
        img = np.expand_dims(img, axis=0)
        try:
          if use_gpu:
            inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
          else:
            inputs = torch.autograd.Variable(torch.from_numpy(img).float())
          d_hist = res_model(inputs)[pick_layer]
          d_hist = d_hist.data.cpu().numpy().flatten()
          d_hist /= np.sum(d_hist)  # normalize
          fin_sample.append({
                          'img':  d_img, 
                          'category':  d_category, 
                          'hist': d_hist
                        })
        except:
          pass
          
    sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #캐시 데이터
    
    pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))
  
    return fin_sample #샘플 데이터

if __name__ == "__main__":
  # evaluate database
  db = database()
  data = db.get_data()
  resnet = ResNetFeat()
  samples = resnet.make_samples(data, 'top')
  print(len(samples))