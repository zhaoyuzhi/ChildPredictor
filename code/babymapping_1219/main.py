''' Baisc packages
'''
import os
import importlib
import logging


''' Configuration packages
'''
import yaml
import argparse
from easydict import EasyDict as edict
from utils import utils_logger
from utils import utils_tb

if __name__=='__main__':
    ''' Parameters
    '''
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, default='./yaml/yaml/Mapping_Xencoder_full_ProGAN_GAN_MSGAN_ACGAN_deepArch_multi-gt_v4.yaml')
    #parser.add_argument('--mode', type=str, default='train', help='which phase you want to set as. train or validation')
    parser.add_argument('--config', type=str, default='./yaml/yaml/validation.yaml')
    parser.add_argument('--mode', type=str, default='validation', help='which phase you want to set as. train or validation')
    cfg = parser.parse_args()
    args = edict(yaml.load(open(cfg.config, 'r'), Loader=yaml.FullLoader))

    ''' 0. Set logger
    '''
    utils_logger.logger_info(args.LOG_CONFIG.logger.name, log_path=args.LOG_CONFIG.logger.logger_path)
    logger = logging.getLogger(args.LOG_CONFIG.logger.name)
    vis_logger = utils_tb.visualboard(args.LOG_CONFIG.tb.tb_path, name=args.NAME, task=args.LOG_CONFIG.tb.task)
    
    ''' 1. Import trainer
	'''
    Trainer = importlib.import_module('{}'.format(args.TRAINER)) 

    #Train
    if cfg.mode=='train':
        Trainer.train(args, logger=logger, vis_logger=vis_logger)
    #Validation
    elif cfg.mode=='validation':
        Trainer.validation(args, logger=logger)
    elif cfg.mode=='randompggan':
        Trainer.pggan_random_samples(args, logger=logger)
    else:
        raise ValueError('Only [train] or [validation] are useful!')

