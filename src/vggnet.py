# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

import pickle
import numpy as np
import cv2
import os

from db import database

# configs for histogram
VGG_model  = 'vgg19'  # model type
pick_layer = 'avg'    # extract feature of this layer
d_type     = 'd1'     # distance type

depth      = 3        # retrieved depth, set to None will count the ap for whole database

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

path = 'C:\\Users\\User\\Documents\\mju\\2023 2학기\\캡스톤 디자인\\CBIR'
os.chdir(path)

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

class VGGNet(VGG):
  def __init__(self, pretrained=True, model='vgg16', requires_grad=False, remove_fc=False, show_params=False):
    super().__init__(make_layers(cfg[model]))
    self.ranges = ranges[model]
    self.fc_ranges = ((0, 2), (2, 5), (5, 7))

    if pretrained:
      exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

    if not requires_grad:
      for param in super().parameters():
        param.requires_grad = False

    if remove_fc:  # delete redundant fully-connected layer params, can save memory
      del self.classifier

    if show_params:
      for name, param in self.named_parameters():
        print(name, param.size())

  def forward(self, x):
    output = {}

    x = self.features(x)

    avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    output['avg'] = avg

    x = x.view(x.size(0), -1)  # flatten()
    dims = x.size(1)
    if dims >= 25088:
      x = x[:, :25088]
      for idx in range(len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x
    else:
      w = self.classifier[0].weight[:, :dims]
      b = self.classifier[0].bias
      x = torch.matmul(x, w.t()) + b
      x = self.classifier[1](x)
      output["fc1"] = x
      for idx in range(1, len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x

    return output

ranges = {
  'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
  'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
  'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
  'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
  'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)


class VGGNetFeat(object):

  def make_samples(self, data, category, verbose=True):
    sample_cache = '{}-{}'.format(VGG_model, pick_layer)
    
    temp = os.path.join(os.getcwd(), 'cache')
    cache_dir = category + ' cache'

    data_img = data['img'].tolist()
    db_img = data['img'][1:].tolist()

    try:
      samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
      samples_img = [sample['img'] for sample in samples]

    except Exception as e: #캐시에 아무 것도 없는
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

        vgg_model = VGGNet(requires_grad=False, model=VGG_model)
        vgg_model.eval()

        if use_gpu:
          vgg_model = vgg_model.cuda()
        
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
          d_hist = vgg_model(inputs)[pick_layer]
          d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
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
  db = database()
  data = db.get_data()
  vggnet = VGGNetFeat()
  samples = vggnet.make_samples(data, 'top')
  print(len(samples))