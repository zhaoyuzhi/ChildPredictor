###########################################################
#　v1 mapping的dataloader
#　返回的是father_img mother_img, child_img
#　对于多个孩子的家庭，会按照孩子的数量组成多个pair
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


''' Some utils
'''
# ----------------------------------------
#            Create image pair
# ----------------------------------------
def get_basic_folder(path):
    # read a folder, return the family folder name list
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            whole_path = os.path.join(root, filespath)
            delete_len = len(whole_path.split('/')[-1]) + 1
            whole_path = whole_path[:-delete_len]
            # only save folder name (one such folder may contain many face images)
            if whole_path not in ret:
                ret.append(whole_path)
    return ret

def get_image_pairs(basic_folder_list):
    # read each folder, return each pair; ret is a 2-dimensional list and the smallest dimension represents pair
    ret = []
    for folder_name in basic_folder_list:
        # for a specific family
        for root, dirs, files in os.walk(folder_name):
            # walk this folder
            for filespath in files:
                if filespath != 'ori.png':
                    # parents
                    if int(filespath[:2]) == 1:
                        father = filespath
                    if int(filespath[:2]) == 2:
                        mother = filespath
            for filespath in files:
                if filespath != 'ori.png':
                    # children, first two integers > 2
                    if int(filespath[:2]) == 3:
                        # temp saves a training / testing pair
                        temp = []
                        temp.append(os.path.join(root, father))     # father
                        temp.append(os.path.join(root, mother))     # mother
                        temp.append(os.path.join(root, filespath))  # children
                        ret.append(temp)
    return ret


class mapping(BaseData):
    def __init__(self, args):
        super(mapping, self).__init__(args)
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
        basic_folder_list = get_basic_folder(scan_dir)
        pair_list = get_image_pairs(basic_folder_list)
    #    print('################## Dataset ##################')
        print('====> 1.Basic_folder_list_num: {}'.format(len(basic_folder_list)))
        print('====> 2.Pair_list_num: {}'.format(len(pair_list)))
        return pair_list


    ''' Customized functions
    '''



    def __len__(self):
        return self.return_len

    def __getitem__(self, idx):
        father_path, mother_path, child_path = self.file_list[idx]

        father_img = self._op_readasTensor(father_path)        #Read as a tensor, CxHxW, value range:0.0-255.0
        father_img = self._op_image_normalize(father_img)                     #normalize to -1,1

        mother_img = self._op_readasTensor(mother_path)        #Read as a tensor, CxHxW, value range:0.0-255.0
        mother_img = self._op_image_normalize(mother_img)                     #normalize to -1,1

        child_img = self._op_readasTensor(child_path)        #Read as a tensor, CxHxW, value range:0.0-255.0
        child_img = self._op_image_normalize(child_img)                     #normalize to -1,1

        return father_img, mother_img, child_img           #img shape:[C,H,W], value:-1~1, imglabel shape:[4] value:0~1






if __name__=='__main__':
    args = edict(yaml.load(open('../yaml/base.yaml', 'r')))
   # args.
    pdata = Example(args.DATASET_CONFIG)

