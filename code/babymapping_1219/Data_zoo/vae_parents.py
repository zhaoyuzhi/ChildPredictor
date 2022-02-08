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

class Vae_parents(BaseData):
    def __init__(self, args):
        super(Vae_parents, self).__init__(args)
        self._op_init_param(args.dataset)      #After this, all parameters defined in yaml can be used.
        self.file_list_father, self.file_list_mother = self._scan_files(args.root_dir, args=args.dataset)


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
        filepath_list_father = []
        filepath_list_mother = []
        filepath_list_child = []
        for root, dirs, files in os.walk(scan_dir):
            for filepath in files:
                if ext in filepath and 'ori' not in filepath:
                    if int(filepath[:2])==1:  #father
                        filepath_list_father.append(os.path.join(root, filepath))
                    elif int(filepath[:2])==2:  #mother
                        filepath_list_mother.append(os.path.join(root, filepath))
                    else:  #child
                        filepath_list_child.append(os.path.join(root, filepath))
        return filepath_list_father, filepath_list_mother                  #Return all files in scan_dir, not include ori.png

    ''' Customized functions
    '''
    def disentangle_label(self, label):
        # input label should be a string
        # for father and mother, the family status is 1, while children is larger than 2
        # gender: 3rd; skin color: 4th; age: 5th; emotion: 6th; glass: 7th; moustache: 8th
        # these are binary labels, and only gender, age, emotion, glass, and moustache are taken into account
        if label[:2] == '01' or label[:2] == '02':
            new_label = str(int(label[:2])) + ',' + label[2] + ',' + label[4] + ',' + label[5] + ',' + label[6] + ',' + label[7]
        else:
            new_label = '0' + ',' + label[2] + ',' + label[4] + ',' + label[5] + ',' + label[6] + ',' + label[7]
        new_label = np.fromstring(new_label, dtype = int, sep = ',')
        new_label = torch.from_numpy(new_label)
        return new_label
    def binarize_label(self, label):
        # index  value  meaning  value  meaning
        #   0      0     child     1     parent
        #   1      0     woman     1      man
        #   2      0     older     1    younger
        #   3      0     smile     1   not smile
        #   4      0     glass     1    no glass
        #   5      0   moustache   1  no moustache
        # Male | Eyeglasses | Mustache | Smiling: 1 = True, 0 = False
        attr = []
        if label[1] == 1:
            attr.append(1)
        if label[1] == 2:
            attr.append(0)
        if label[4] == 1:
            attr.append(1)
        if label[4] == 2:
            attr.append(0)
        if label[5] == 1:
            attr.append(1)
        if label[5] == 2:
            attr.append(0)
        if label[3] == 2:
            attr.append(1)
        if label[3] == 1 or label[3] == 3:
            attr.append(0)
        attr = np.array(attr)
        attr = torch.from_numpy(attr).float()
        assert attr.shape[0]==4, '{}'.format(label)
        return attr


    def __len__(self):
        return min(len(self.file_list_father), len(self.file_list_mother))

    def __getitem__(self, idx):
        img_father = self._op_readasTensor(self.file_list_father[idx])        #Read as a tensor, CxHxW, value range:0.0-255.0
        img_mother = self._op_readasTensor(self.file_list_mother[idx])        #Read as a tensor, CxHxW, value range:0.0-255.0

        img_father = self._op_image_normalize(img_father)                     #normalize to -1,1
        img_mother = self._op_image_normalize(img_mother)                     #normalize to -1,1

        imglabel_father = self.file_list_father[idx].split('/')[-1][:-4]      #'01122222'
        imglabel_mother = self.file_list_mother[idx].split('/')[-1][:-4]      #'02122222'


        imglabel_father = self.disentangle_label(imglabel_father)
        imglabel_mother = self.disentangle_label(imglabel_mother)

        imglabel_father = self.binarize_label(imglabel_father)
        imglabel_mother = self.binarize_label(imglabel_mother)

        return img_father, img_mother, imglabel_father, imglabel_mother       #img shape:[C,H,W], value:-1~1, imglabel shape:[4]










if __name__=='__main__':
    args = edict(yaml.load(open('../yaml/base.yaml', 'r')))
   # args.
    pdata = Example(args.DATASET_CONFIG)

