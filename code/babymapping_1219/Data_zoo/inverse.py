###########################################################
#　v1 inverse的dataload
###########################################################
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

class inverse(BaseData):
    def __init__(self, args):
        super(inverse, self).__init__(args)
        self._op_init_param(args.dataset)      #After this, all parameters defined in yaml can be used.
        self.file_list = self._scan_files(args.root_dir, args=args.dataset)
        self.return_len = int(len(self.file_list))

    def _op_init_param(self, args_d):
        ''' Rewrite or use default. This func is to get more parameters 
            that belong to args.DATASEt_CONFIG. Uncomment one to select
            another one.
        '''
        # Style 1: Use default method defined by father class.
        super()._op_init_param(args_d)
        # Style 2: Rewrite by yourself
        #self.xxx = 'xxx'

    ''' Image normalize'''
    def _op_image_normalize(self, img, max_value=255.):
        '''
            input: tensor, value:0.0-255.0
            output: tensor, value:0.0-1.0
            This function alse can be rewrite by user to have a customized value scope.
        '''
        img = img.div(max_value)   #0-1
        img = img.mul(2.0).add(-1.0)   #0-2 -> -1-1
        return img

    def _op_readasTensor(self, path):
        ''' Rewrite or use default. This func is to get more parameters 
            that belong to args.DATASEt_CONFIG. Uncomment one to select
            another one.
        '''
        # Style 1: Use default method defined by father class.
        img = super()._op_readasTensor(path)

        assert img.shape[0]==3, '{}'.format(path)
        return img
        # Style 2: Rewrite by yourself

    def _scan_files(self, scan_dir, args=None)->list:
        ext = args.ext
        phase = args.phase
        scan_dir = os.path.join(scan_dir, phase)
        assert os.path.isdir(scan_dir)
        filepath_list = []
        for root, dirs, files in os.walk(scan_dir):
            for filepath in files:
                if ext in filepath and 'ori' not in filepath:
                    if int(filepath[:2])!=1 and int(filepath[:2])!=2:  #only child can be detected.
                        filepath_list.append(os.path.join(root, filepath))
        return filepath_list                  #Return all files in scan_dir, not include ori.png





    ''' Customized functions
    '''



    def __len__(self):
        return self.return_len

    def __getitem__(self, idx):
        imgitem = self.file_list[idx]

        img = self._op_readasTensor(imgitem)        #Read as a tensor, CxHxW, value range:0.0-255.0
        img = self._op_image_normalize(img)                     #normalize to -1,1

        return img           #img shape:[C,H,W], value:-1~1, imglabel shape:[4] value:0~1






if __name__=='__main__':
    args = edict(yaml.load(open('../yaml/base.yaml', 'r')))
   # args.
    pdata = Example(args.DATASET_CONFIG)

