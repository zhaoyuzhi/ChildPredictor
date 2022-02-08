''' Baisc packages
'''
import os
import glob
import tqdm
import copy
import random
import importlib
import numpy as np
import cv2

for ij in range(16):
    a = ij
    b = bin(a)[-4:]
    print(b)
