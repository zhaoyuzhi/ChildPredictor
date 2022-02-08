import os
import glob
import imageio
import cv2
import torch
import numpy as np
import torch.utils.data as torchdata



class BaseData(torchdata.Dataset):
    def __init__(self, args):
        super(BaseData, self).__init__()
        self.root_dir = args.root_dir            #Root dir of dataset you want to load
        self.glob_mode = args.glob_mode          #how to glob files, now it's unuseful.
        self.read_mode = args.read_mode          #use which func to load img as a numpy matrix, cv2 or imageio?
        self._op_init_param(args.dataset)        #init other parameters in args.DATASET_DATA.dataset so that you can use self.xxx etc.

    ''' Init parameters'''
    def _op_init_param(self, args_d):
        # Can be rewrite.
        for key, value in args_d.items():
            setattr(self, key, value)

    ''' Read image'''
    def _op_read_image(self, path):
        ''' 
            input -> path of image
            output: A numpy matrix, 
                    shape: CxHxW, 
                    value: 0-255,
                    type: uint8
        '''
        if self.read_mode == 'imageio':
            img = imageio.imread(path)
            assert img.ndim == 3
            img = np.ascontiguousarray(img.transpose((2,0,1)))     
        elif self.read_mode == 'cv2':
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)            # BGR or G
            assert img.ndim == 3
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.ascontiguousarray(img.transpose((2,0,1)))      #BGR to RGB
        else:
            raise ValueError('Wrong read mode, <imageio> and <cv2> are available, you use <{}>'.format(self.read_mode))
        return img

    ''' Transfer to tensor'''
    def _op_to_tensor(self, img):
        '''
            input: numpy matrix
            output: tensor
                    shape: CxHxW
                    value: 0.0-255.0(float)
        '''
        img = img.astype(np.float32)
        img = torch.from_numpy(img).float()
        return img

    ''' Image normalize'''
    def _op_image_normalize(self, img, max_value=255.):
        '''
            input: tensor, value:0.0-255.0
            output: tensor, value:0.0-1.0
            This function alse can be rewrite by user to have a customized value scope.
        '''
        img = img.div(max_value)
        return img

    ''' A pipline: read -> transform -> normalize'''
    def _op_readasTensor(self, path):
        '''
            Input: path
            Return: tensor, value:0.0-255.0
            Should be implemented by user.
        '''
        img = self._op_read_image(path)
        img = self._op_to_tensor(img)

        return img

    ''' Get file list'''
    def _scan_files(self, scan_dir, args=None):
        raise NotImplementedError(f'Should be implemented in derived class!')


