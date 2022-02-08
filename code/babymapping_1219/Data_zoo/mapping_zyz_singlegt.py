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
import torchvision
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

def get_image_pairs(basic_folder_list, out_num):
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
            # temp saves a training / testing pair
            temp = []
            temp.append(os.path.join(root, father))     # father
            temp.append(os.path.join(root, mother))     # mother
            for filespath in files:
                if filespath != 'ori.png':
                    # children, first two integers > 2
                    if int(filespath[:2]) > 2 and int(filespath[:2]) < (2 + out_num):
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
        out_num = args.out_num
        scan_dir = os.path.join(scan_dir, phase)
        assert os.path.isdir(scan_dir)
        basic_folder_list = get_basic_folder(scan_dir)
        pair_list = get_image_pairs(basic_folder_list, out_num)
    #    print('################## Dataset ##################')
        print('====> 1.Basic_folder_list_num: {}'.format(len(basic_folder_list)))
        print('====> 2.Pair_list_num: {}'.format(len(pair_list)))
        return pair_list


    ''' Customized functions
    '''
    ''' Customized functions
    '''
    def get_label(self, img_path):
        name = img_path.split('/')[-1].split('.')[0]
        attr = []
        if int(name[2])==2:
            attr.append(0.0)
        else:
            attr.append(1.0)
        if int(name[6])==2:
            attr.append(0.0)
        else:
            attr.append(1.0)
        if int(name[5])==2:
            attr.append(1.0)
        else:
            attr.append(0.0)
        if int(name[3])==3:
            attr.append(1.0)
        else:
            attr.append(0.0)

        attr = np.array(attr)
        attr = torch.from_numpy(attr).float()
        assert attr.shape[0]==4, '{}'.format(img_path)
        return attr

    def op_hflip(self, img):
        img = torchvision.transforms.functional.to_pil_image(img)
        img = torchvision.transforms.functional.hflip(img)
        img = torchvision.transforms.functional.to_tensor(img)        
        return img

    def __len__(self):
        return self.return_len

    def __getitem__(self, idx):
        combine_path = self.file_list[idx]
        if len(combine_path) < 3:
            print(combine_path)
        child_img_list = []
        child_label = []
        combine_path = sorted(combine_path)
       
        for face_id in range(len(combine_path)):
            img_path = combine_path[face_id]
            faceidx = int(img_path.split('/')[-1][:2])
            img = self._op_readasTensor(img_path)        #Read as a tensor, CxHxW, value range:0.0-255.0
            img = self._op_image_normalize(img)                     #normalize to -1,1

            if faceidx == 1:                                # father
                father_img = img
            if faceidx == 2:                                # mother
                mother_img = img
            if faceidx == 3:       # >2
                assert img.shape[0] == 3
                child_img_list.append(img)
                child_img_list.append(img)
                child_img_list.append(img)
                child_img_list.append(img)
                child_label.append(self.get_label(img_path))
                child_label.append(self.get_label(img_path))
                child_label.append(self.get_label(img_path))
                child_label.append(self.get_label(img_path))
            #if faceidx == 4:                  #存在04号孩子
              #  assert len(child_img_list) == 1
            #    assert img.shape[0] == 3
            #    child_img_list.append(img)
            #    child_label.append(self.get_label(img_path))
            #if faceidx == 5:                  #存在05号孩子
            #    assert img.shape[0] == 3
            #    child_img_list.append(img)
            #    child_label.append(self.get_label(img_path))
            #if faceidx == 6:                  #存在06号孩子
            #    assert img.shape[0] == 3
            #    child_img_list.append(img)
            #    child_label.append(self.get_label(img_path))
            #if face_id == len(combine_path)-1 and len(child_img_list)==1:    #只有03号孩子，此时需要将03号进行翻转得到04,06号孩子，05号与03号相同
            #    img_aug = child_img_list[0]
            #    img_aug = self.op_hflip(img_aug)
            #    child_img_list.append(img_aug)         #04号为翻转的
            #    child_img_list.append(child_img_list[0])  #05号为03号一致
            #    child_img_list.append(img_aug)         #06号为翻转的
            #    child_label.append(child_label[0])
            #    child_label.append(child_label[0])
            #    child_label.append(child_label[0])
            #elif face_id == len(combine_path)-1 and len(child_img_list)==2:   #有03号和04号孩子，此时需要将03,04分别翻转，得到05和06号孩子
            #    img_aug1 = self.op_hflip(child_img_list[0])
            #    img_aug2 = self.op_hflip(child_img_list[1])
            #    child_img_list.append(img_aug1)
            #    child_img_list.append(img_aug2)
            #    child_label.append(child_label[0])
            #    child_label.append(child_label[1])
            #elif face_id == len(combine_path)-1 and len(child_img_list)==3:   #有03,04和05号孩子，此时需要将05翻转，得到06号孩子
            #    img_aug = self.op_hflip(child_img_list[2])
            #    child_img_list.append(img_aug)
            #    child_label.append(child_label[2])
        assert not isinstance(child_label, int), "{}".format(combine_path)
        return father_img, mother_img, child_img_list, child_label       #img shape:[C,H,W], value:-1~1, imglabel shape:[4] value:0~1


if __name__=='__main__':
    args = edict(yaml.load(open('../yaml/base.yaml', 'r')))
   # args.
    pdata = Example(args.DATASET_CONFIG)

