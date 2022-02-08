import os
import math
import torch
import torchvision.utils as t_utils
from tensorboardX import SummaryWriter


class visualboard(object):
    def __init__(self, save_path, name='base', task='task1'):
        save_path = os.path.join(save_path, name, task)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.writer = SummaryWriter(save_path)

    def visual_text(self, loss_dict:dict, epoch:int, main_tag='Loss'):
        """Follow the previous THREE split visualization fashion"""
        for loss, value in loss_dict.items():  
            self.writer.add_scalar("{}/item-{}".format(main_tag, loss), value, epoch)

    def visual_image(self, image_dict:dict, epochs:int, normalize=False, main_tag='visual', num_row=4):
        for name, img in image_dict.items():
            image_show = t_utils.make_grid(img, nrow=num_row, normalize=normalize, range=(-1, 1))
            if '-' in name:
                name_list = name.split('-')
                assert len(name_list)==2              #Should be len=2, 0 for tag, 1 for name
                self.writer.add_image('{}-{}/{}'.format(main_tag, name_list[0], name_list[1]), image_show, global_step=epochs)
            else:   
                self.writer.add_image('{}/{}'.format(main_tag, name), image_show, global_step=epochs)

    def single_image(self, text, images, normalize=False):
        image_show = t_utils.make_grid(torch.cat((images[0], images[1], images[2], images[3], images[4], images[5]), dim=0),
                nrow=len(images)//2, normalize=normalize, range=(-1,1))
        
        self.writer.add_image(text, image_show, global_step=0)

    def visual_eval_for_quant(self, eval_dict, iters):
        self.writer.add_scalars("eval_quant", eval_dict, iters)
