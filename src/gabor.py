# -*- coding: utf-8 -*-

from __future__ import print_function

from db import database

from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi

import multiprocessing

import pickle
import numpy as np
import cv2
import os


theta     = 4
frequency = (0.1, 0.5, 0.8)
sigma     = (1, 3, 5)
bandwidth = (0.3, 0.7, 1)

n_slice  = 2
h_type   = 'global'
d_type   = 'cosine'

depth    = 1

def make_gabor_kernel(theta, frequency, sigma, bandwidth):
  kernels = []
  for t in range(theta):
    t = t / float(theta) * np.pi
    for f in frequency:
      if sigma:
        for s in sigma:
          kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
          kernels.append(kernel)
      if bandwidth:
        for b in bandwidth:
          kernel = gabor_kernel(f, theta=t, bandwidth=b)
          kernels.append(kernel)
  return kernels

gabor_kernels = make_gabor_kernel(theta, frequency, sigma, bandwidth)
if sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(sigma), "kernel nums error in make_gabor_kernel()"
elif not sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(bandwidth), "kernel nums error in make_gabor_kernel()"
elif sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * (len(sigma) + len(bandwidth)), "kernel nums error in make_gabor_kernel()"
elif not sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency), "kernel nums error in make_gabor_kernel()"

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

class Gabor(object):  
  
  def gabor_histogram(self, img, type=h_type, n_slice=n_slice, normalize=True):
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._gabor(img, kernels=gabor_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, len(gabor_kernels)))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._gabor(img_r, kernels=gabor_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _feats(self, image, kernel):
    feats = np.zeros(2, dtype=np.double)
    filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
    feats[0] = filtered.mean()
    feats[1] = filtered.var()
    return feats
  
  
  def _power(self, image, kernel):
    image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
    f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    feats = np.zeros(2, dtype=np.double)
    feats[0] = f_img.mean()
    feats[1] = f_img.var()
    return feats
  
  
  def _gabor(self, image, kernels=make_gabor_kernel(theta, frequency, sigma, bandwidth), normalize=True):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
  
    img = color.rgb2gray(image)
  
    results = []
    feat_fn = self._power
    for kernel in kernels:
      results.append(pool.apply_async(self._worker, (img, kernel, feat_fn)))
    pool.close()
    pool.join()
    
    hist = np.array([res.get() for res in results])
  
    if normalize:
      hist = hist / np.sum(hist, axis=0)
  
    return hist.T.flatten()
  
  def _worker(self, img, kernel, feat_fn):
    try:
      ret = feat_fn(img, kernel)
    except:
      print("return zero")
      ret = np.zeros(2)
    return ret
  
  def make_samples(self, data, category, verbose=True):
    sample_cache = "gabor-{}-n_slice{}-theta{}-frequency{}-sigma{}-bandwidth{}".format(h_type, n_slice, theta, frequency, sigma, bandwidth)

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
        img = cv2.imread(d_img)
        d_hist = self.gabor_histogram(img, type=h_type, n_slice=n_slice)

        fin_sample.append({
                        'img':  d_img, 
                        'category':  d_category, 
                        'hist': d_hist
                      })

    sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #캐시 데이터

    pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))

    return fin_sample #샘플 데이터


if __name__ == "__main__":
  db = database()
  data = db.get_data()
  gabor = Gabor()
  samples = gabor.make_samples(data, 'top')
  print(len(samples))