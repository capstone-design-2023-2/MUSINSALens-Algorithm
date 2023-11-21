# -*- coding: utf-8 -*-

from __future__ import print_function

from db import database

from skimage.feature import daisy
from skimage import color

import pickle
import numpy as np
import imageio
import math

import os


n_slice    = 2
n_orient   = 8
step       = 10
radius     = 30
rings      = 2
histograms = 6
h_type     = 'region'
d_type     = 'd1'

depth      = 3

R = (rings * histograms + 1) * n_orient

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

class Daisy(object):

  def histogram(self, input, type=h_type, n_slice=n_slice, normalize=True):
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    P = math.ceil((height - radius*2) / step) 
    Q = math.ceil((width - radius*2) / step)
    assert P > 0 and Q > 0, "input image size need to pass this check"
  
    if type == 'global':
      hist = self._daisy(img)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, R))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._daisy(img_r)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _daisy(self, img, normalize=True):
    image = color.rgb2gray(img)
    descs = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=n_orient)
    descs = descs.reshape(-1, R)  # shape=(N, R)
    hist  = np.mean(descs, axis=0)  # shape=(R,)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist
  
  
  def make_samples(self, data, category, verbose=True):
    sample_cache = "daisy-{}-n_slice{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(h_type, n_slice, n_orient, step, radius, rings, histograms)
  
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

        d_img, d_category = getattr(d, "img"), getattr(d, "category")
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        fin_sample.append({
                        'img':  d_img, 
                        'category':  d_category, 
                        'hist': d_hist
                      })
      pickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #캐시 데이터

    pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))
  
    return fin_sample #샘플 데이터


if __name__ == "__main__":
  db = database()
  data = db.get_data()
  daisy = Daisy()
  sample = daisy.make_samples(data, 'top')
  print(len(sample))

  