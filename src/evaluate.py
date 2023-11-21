from __future__ import print_function

import numpy as np
from scipy import spatial
import cv2

from db import database

class Evaluation(object):

  def make_samples(self):
    raise NotImplementedError("Needs to implemented this method")

