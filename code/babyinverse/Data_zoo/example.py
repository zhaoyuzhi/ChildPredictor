import os
import glob
import imageio
import cv2
import torch
import numpy as np
#from base import BaseData               #1
from Data_zoo.base import BaseData     #2

import yaml
from easydict import EasyDict as edict

class Example(BaseData):
    def __init__(self, args):
        super(Example, self).__init__(args)
        self._op_init_param(args.dataset)      #After this, all parameters defined in yaml can be used.
        img = self._op_readasTensor('./IMG_20200527_195455.jpg')
        print(img[0,0,0])

    def _op_init_param(self, args_d):
        ''' Rewrite or use default. This func is to get more parameters 
            that belong to args.DATASEt_CONFIG. Uncomment one to select
            another one.
        '''
        # Style 1: Use default method defined by father class.
        super()._op_init_param(args_d)
        # Style 2: Rewrite by yourself
        #self.xxx = 'xxx'

    def _op_readasTensor(self, path):
        ''' Rewrite or use default. This func is to get more parameters 
            that belong to args.DATASEt_CONFIG. Uncomment one to select
            another one.
        '''
        # Style 1: Use default method defined by father class.
        return super()._op_readasTensor(path)
        # Style 2: Rewrite by yourself

    def _scan_files(self, scan_dir, args=None):
        pass

    ''' Customized functions
    '''


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass








if __name__=='__main__':
    args = edict(yaml.load(open('../yaml/base.yaml', 'r')))
   # args.
    pdata = Example(args.DATASET_CONFIG)

