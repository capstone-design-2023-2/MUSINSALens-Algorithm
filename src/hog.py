from __future__ import print_function

from skimage.feature import hog
from skimage import color
import pickle
import numpy as np
import cv2
import os

from db import database

n_bin    = 10
n_slice  = 6
n_orient = 8
p_p_c    = (2, 2)
c_p_b    = (1, 1)
h_type   = 'region'
d_type   = 'd1'
depth    = 5

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

class HistogramOfGradients(object):
  def histogram(self, img, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._HOG(img, n_bin)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._HOG(img_r, n_bin)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()

  def _HOG(self, img, n_bin, normalize=True):
    image = color.rgb2gray(img)
    fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
    bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
    hist, _ = np.histogram(fd, bins=bins)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist

  def make_samples(self, data, category, verbose=True):
    sample_cache = "HOG-{}-n_bin{}-n_slice{}-n_orient{}-ppc{}-cpb{}".format(h_type, n_bin, n_slice, n_orient, p_p_c, c_p_b)
  
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
        sample['hist'] = np.float32(sample['hist'])
        fin_sample.append(sample)
      else:
        for item in data.itertuples():
          if item.img == img:
            d = item
            break

        d_img, d_category = getattr(d, "img"), getattr(d, "category")
        img = cv2.imread(d_img)
        d_hist = self.histogram(img, type=h_type, n_slice=n_slice)

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
  hog_instance = HistogramOfGradients()
  samples = hog_instance.make_samples(data, 'top')
  print(len(samples))