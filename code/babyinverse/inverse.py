''' Baisc packages
'''
import os
import glob
import tqdm
import copy
import random
import importlib
import numpy as np

''' Configuration packages
'''
import yaml
import argparse
from easydict import EasyDict as edict

''' PyTorch packages
'''
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.functional import cosine_similarity

from torch.utils.data import DataLoader
from utils.utils_check import check_var
from utils.utils_loss import VGGLoss



def train(args, logger=None, vis_logger=None):
    logger.info('=> Import libs')
    Datalib     = importlib.import_module('.{}'.format(args.DATASET_CONFIG.dataset.name), package=args.DATASET_CONFIG.package)
    Model       = importlib.import_module('.{}'.format(args.MODEL.name[0]), package=args.MODEL.package)
    Model_PGGAN = importlib.import_module('.{}'.format(args.MODEL.name[1]), package=args.MODEL.package)



    logger.info('=> Set dataloader')
    trainset = Datalib.inverse(args.DATASET_CONFIG)
    train_loader = DataLoader(trainset, batch_size=args.TRAIN.batch_size, num_workers=args.TRAIN.num_workers, drop_last=True, pin_memory=True, shuffle=True)


    logger.info('=> Make models and load checkpoint')
    encoder_net = Model.Vgg16(pre_train=args.MODEL.vgg.pre_train,
                              requires_grad=args.MODEL.vgg.requires_grad,
                              vae_encoder=args.MODEL.vgg.use_vae,
                              global_pooling=args.MODEL.vgg.global_pooling,
                              if_downsample=args.MODEL.vgg.if_downsample)
    G_net = Model_PGGAN.PGGANGenerator(model_name=args.MODEL.pggan.model_name, logger=logger)
    D_net = Model.Discriminator(size=args.MODEL.D.imgsize,
                                input_channel=args.MODEL.D.input_channel,
                                ndf=args.MODEL.D.ndf,
                                channel_multiplier=args.MODEL.D.channel_multiplier,
                                use_sigmoid=args.MODEL.D.use_sigmoid,
                                use_sn=args.MODEL.D.use_sn)
    if args.USE_GPU:
        logger.info('===> Use {} GPUs'.format(args.NUM_GPU))
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        GPU_list = [i for i in range(args.NUM_GPU)]
        encoder_net = nn.DataParallel(encoder_net, device_ids=GPU_list)
        D_net = nn.DataParallel(D_net, device_ids=GPU_list)
        encoder_net.to(device)
        D_net.to(device)
    else:
        ValueError('Unsupported mode!')
    if args.MODEL.checkpoint.ckp_flag:
        logger.info('===> Load ckp for <vgg_encoder> from {}'.format(args.MODEL.checkpoint.ckp_path))
        statedict = torch.load(args.MODEL.checkpoint.ckp_path)
        encoder_net.module.load_state_dict(statedict)
    

    logger.info('=> Set loss')
    MSE_loss = nn.MSELoss()
   # MSE_loss = nn.L1Loss()
    D_loss = nn.BCELoss()
    VGG_loss = VGGLoss(weight_vgg=1.)
    if args.USE_GPU:
        MSE_loss.to(device)
        VGG_loss.to(device)
        D_loss.to(device)
    
    
    logger.info('=> Set optimizer')
    optimizer_encoder = torch.optim.Adam(encoder_net.module.parameters(), \
                        lr=args.OPTIMIZER.lr_encoder, \
                        betas=(args.OPTIMIZER.Adam.beta1, args.OPTIMIZER.Adam.beta2), \
                        weight_decay=1e-4)
    optimizer_D       = torch.optim.Adam(D_net.module.parameters(), \
                        lr=args.OPTIMIZER.lr_D, \
                        betas=(args.OPTIMIZER.Adam.beta1, args.OPTIMIZER.Adam.beta2), \
                        weight_decay=1e-4)

    """ Intrisic function"""
    def cor_square_error(x, y, eps=1e-12):
        return (1.0 - cosine_similarity(x, y, eps=eps)).mean()
    def save_checkpoint(net, name, dir_path, epoch, bs, gap=1):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if epoch % gap == 0:
            path = os.path.join(dir_path, '{}_Batchsize_{}_Epoch_{}.pth'.format(name, bs, epoch))
            torch.save(net.module.state_dict(), path)
            logger.info('===> Save checkpoint at {}'.format(path))

    logger.info('=> Begin to train the model')
    encoder_net.train()
    D_net.train()
    Tensor = torch.cuda.FloatTensor if args.USE_GPU else torch.FloatTensor
   
    for epoch in range(0, args.TRAIN.epochs):
        bar = tqdm.tqdm(train_loader)
        dis_loss_list = []
        gan_loss_list = []
        latent_loss_list = []
        en_mse_loss_list = []
        en_vgg_loss_list = []
        encoder_loss_list = []
       ####################################################
       #       gt_img
       #         |
       #     inverse_net
       #         |
       #       lat_code  + label +  uniform
       #                    |
       #                  pggan
       #                    |
       #                recon_img
       #                    |
       #                inverse_net
       #                    |
       #                 rec_code
       ####################################################
        for idx, (input_img, input_label) in enumerate(bar):
            input_img = input_img.cuda()
            input_label = input_label.cuda()
            uniform_code = torch.Tensor(input_img.shape[0], args.TRAIN.uniform_dim).uniform_().type_as(input_img)
            uniform_code = uniform_code.cuda()

            latent_code4, skip = encoder_net(input_img)  #输入为gt，得到其inverse后的code
            latent_code = torch.cat((latent_code4, input_label, uniform_code), dim=1)
            recon_img = G_net.synthesize(latent_code)  #输入为pred的code，通过pggan得到recon_img，希望它可以和gt尽可能一样

            #Update D_net
            optimizer_D.zero_grad()
            pred_real = D_net(input_img)
            pred_fake = D_net(recon_img.detach())
            dis_loss = 0

            for idx_real, idx_fake in zip(pred_real, pred_fake):
                global_real_label = Tensor(idx_real.size()).fill_(1.).cuda()
                global_fake_label = Tensor(idx_fake.size()).fill_(0.).cuda()
                dis_loss += (D_loss(idx_real, global_real_label) + D_loss(idx_fake, global_fake_label))
            dis_loss.backward()
            optimizer_D.step()
            dis_loss_list.append(dis_loss.item())
            
            #Update encoder_net
            optimizer_encoder.zero_grad()
            pred_fake = D_net(recon_img)
            recon_latent_code, _ = encoder_net(recon_img)
            
            gan_loss = 0  #gan loss
            for idx_fake in pred_fake:
                global_real_label = Tensor(idx_fake.size()).fill_(1.).cuda()
                gan_loss += D_loss(idx_fake, global_real_label)
            latent_loss = 1. * cor_square_error(recon_latent_code, latent_code4) +\
                          1. * MSE_loss(recon_latent_code, latent_code4)
            mse_loss = MSE_loss(recon_img, input_img)
            vgg_loss = VGG_loss(recon_img, input_img)
            encoder_loss = (args.TRAIN.weight_mse * mse_loss +\
                           args.TRAIN.weight_vgg * vgg_loss +\
                           args.TRAIN.weight_latent * latent_loss +\
                           args.TRAIN.weight_gan * gan_loss).mean()
            encoder_loss.backward()
            optimizer_encoder.step()
            gan_loss_list.append(gan_loss.item())
            latent_loss_list.append(latent_loss.item())
            en_mse_loss_list.append(mse_loss.item())
            en_vgg_loss_list.append(vgg_loss.item())
            encoder_loss_list.append(encoder_loss.item())
            

            if idx % args.TRAIN.save_iter == 0:
                #dump images to tensorboard
                vis_logger.visual_image(
                    {'input_image': input_img,
                     'recon_image': recon_img},
                    (idx+1), normalize=True
                )

        vis_logger.visual_text(
            {'D_net': np.mean(dis_loss_list)},
            epoch,
            main_tag='D_loss'
        )
        vis_logger.visual_text(
            {'EncoderNet': np.mean(encoder_loss_list),
             'gan_loss': np.mean(gan_loss_list),
             'latent_loss': np.mean(latent_loss_list),
             'recon_loss': np.mean(en_mse_loss_list),
             'perceptual_loss': np.mean(en_vgg_loss_list)},
            epoch,
            main_tag='Encoder_loss'
        )
        logger.info('[Epoch: {0}/{1}] | [D_net: {2:.5f}] [EncoderNet: {3:.5f}] [gan_loss: {4:.5f}] [latent_loss: {5:.5f}] [recon_loss: {6:.5f}] [perceptual_loss: {7:.5f}]'.format(
            epoch,
            args.TRAIN.epochs,
            np.mean(dis_loss_list),
            np.mean(encoder_loss_list),
            np.mean(gan_loss_list),
            np.mean(latent_loss_list),
            np.mean(en_mse_loss_list),
            np.mean(en_vgg_loss_list)
        ))

        save_checkpoint(encoder_net, 'EncoderNet', args.MODEL.checkpoint.save_path[0], epoch, args.TRAIN.batch_size, gap=2)
        save_checkpoint(encoder_net, 'DNet', args.MODEL.checkpoint.save_path[1], epoch, args.TRAIN.batch_size, gap=2)

            
