# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools
import os
import numpy as np
import pickle
from scipy import spatial
import cv2

from db import database

# cache dir - category 별로 캐시를 다르게 만들기
categories = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 'short sleeve dress', 'short sleeve outwear',
              'short sleeve top', 'shorts', 'skirt', 'sling dress', 'sling', 'trousers', 'vest dress', 'vest', 'top']

os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

def histogram(input, normalize = True):
    img = cv2.imread(input)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    if normalize:
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    return hist
    
class Color(object):

    def make_samples(self, data, category, verbose = True):
        sample_cache = 'color'
        
        temp = os.path.join(os.getcwd(), 'cache')
        cache_dir = category + ' cache'

        data_img = data['img'].tolist()
        db_img = data['img'][1:].tolist()

        try:
            samples = pickle.load(open(temp, cache_dir, sample_cache, "rb", True))
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
                d_hist = histogram(d_img)
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
    color = Color()
    samples = color.make_samples(data, 'top')
    print(len(samples))