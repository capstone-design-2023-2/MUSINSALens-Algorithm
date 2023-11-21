# pylint: disable=invalid-name,missing-docstring,exec-used,too-many-arguments,too-few-public-methods,no-self-use
from __future__ import print_function

import sys
import cv2
import numpy as np
import pandas as pd
import json

from color import Color
from db import database
from gabor import Gabor
from hog import HistogramOfGradients
from resnet import ResNetFeat
from vggnet import VGGNetFeat

query_idx = -1

if __name__ == "__main__":
  query_json = open('C:\\Users\\User\\Documents\\mju\\2023 2학기\\캡스톤 디자인\\CBIR\\query.json', encoding = 'utf-8')
  query = json.load(query_json)
  query = pd.DataFrame([query])
  query_category = query['category'][0]

  db = database()
  data = db.get_data()

  #category limit
  data_category = data[data['category'] == query_category]
  data_category = pd.concat([query, data_category], ignore_index = True)

  feature_extract_methods = {
      "color": Color, #색상
      "hog": HistogramOfGradients, #형태
      "gabor": Gabor, #질감
      "vgg": VGGNetFeat,
      "resnet": ResNetFeat
  }

  # try:
  #     mthd = sys.argv[1].lower()
  # except IndexError:
  #     print("usage: {} <method>".format(sys.argv[0]))
  #     print("supported methods:\ncolor, daisy, edge, gabor, hog, vgg, resnet")

  #     sys.exit(1)

  for mthd in feature_extract_methods.keys() :
    print(mthd)

    samples = getattr(feature_extract_methods[mthd](), "make_samples")(data_category, query_category)
    
    #img_path = [item['img'] for item in samples]
    hists = [item['hist'] for item in samples]

    #img_hists = {'img': img_path, 'hists': hists}

    query = hists[0]

    query = query.astype(np.float32)
    hists = [hist.astype(np.float32) for hist in hists]

    method = cv2.HISTCMP_BHATTACHARYYA

    for i, histogram in enumerate(hists):
      ret = cv2.compareHist(query, histogram, method)
      print("img%d :%7.2f" % (i + 1, ret), end='\t')
    print()