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


def validation(args, logger=None, vis_logger=None):
    logger.info('********************* Validation Phase ************************')
    logger.info('=> Import libs')
    Datalib       = importlib.import_module('.{}'.format(args.DATASET_CONFIG.dataset.name), package=args.DATASET_CONFIG.package)
    Model         = importlib.import_module('.{}'.format(args.MODEL.name[0]), package=args.MODEL.package)
    Model_Encoder = importlib.import_module('.{}'.format(args.MODEL.name[1]), package=args.MODEL.package)
    Model_AttGAN  = importlib.import_module('.{}'.format(args.MODEL.name[2]), package=args.MODEL.package)
    Model_PGGAN   = importlib.import_module('.{}'.format(args.MODEL.name[3]), package=args.MODEL.package)



def train(args, logger=None, vis_logger=None):
    logger.info('=> Import libs')
    Datalib       = importlib.import_module('.{}'.format(args.DATASET_CONFIG.dataset.name), package=args.DATASET_CONFIG.package)
    Model         = importlib.import_module('.{}'.format(args.MODEL.name[0]), package=args.MODEL.package)
    Model_Encoder = importlib.import_module('.{}'.format(args.MODEL.name[1]), package=args.MODEL.package)
    Model_AttGAN  = importlib.import_module('.{}'.format(args.MODEL.name[2]), package=args.MODEL.package)
    Model_PGGAN   = importlib.import_module('.{}'.format(args.MODEL.name[3]), package=args.MODEL.package)

    logger.info('=> Set dataloader')
    trainset     = Datalib.mapping(args.DATASET_CONFIG)  #返回一组父母img和孩子img，总数同孩子数
    train_loader = DataLoader(trainset, batch_size=args.TRAIN.batch_size, num_workers=args.TRAIN.num_workers, drop_last=True, pin_memory=True, shuffle=True)

    
    logger.info('=> Make models and load checkpoint')
    Encoder_net = Model_Encoder.Vgg16(pre_train=args.MODEL.vgg.pre_train,
                                      requires_grad=args.MODEL.vgg.requires_grad,
                                      vae_encoder=args.MODEL.vgg.use_vae,
                                      global_pooling=args.MODEL.vgg.global_pooling,
                                      if_downsample=args.MODEL.vgg.if_downsample)
    Mapping_net = Model.MappingNet(args.MODEL.mapping.in_channels, args.MODEL.mapping.out_channels, args.MODEL.mapping.out_num, args.MODEL.mapping.input_norm)
    AttGAN_net  = Model_AttGAN.Generator(
                                    args.MODEL.attgan.enc_dim, args.MODEL.attgan.enc_layers, args.MODEL.attgan.enc_norm, args.MODEL.attgan.enc_acti,
                                    args.MODEL.attgan.dec_dim, args.MODEL.attgan.dec_layers, args.MODEL.attgan.dec_norm, args.MODEL.attgan.dec_acti,
                                    args.MODEL.attgan.n_attrs, args.MODEL.attgan.shortcut_layers, args.MODEL.attgan.inject_layers, args.MODEL.attgan.img_size)
    G_net = Model_PGGAN.PGGANGenerator(model_name=args.MODEL.pggan.model_name, logger=logger)
    if args.USE_GPU:
        logger.info('===> Use {} GPUs'.format(args.NUM_GPU))
        assert torch.cuda.is_available()
        device      = torch.device('cuda')
        GPU_list    = [i for i in range(args.NUM_GPU)]
        Mapping_net = nn.DataParallel(Mapping_net, device_ids=GPU_list)
        Encoder_net = nn.DataParallel(Encoder_net, device_ids=GPU_list)
        AttGAN_net  = nn.DataParallel(AttGAN_net, device_ids=GPU_list)
        Mapping_net.to(device)
        Encoder_net.to(device)
        AttGAN_net.to(device)
        G_net.model.to(device)
    else:
        ValueError('Unsupported mode!')
    if args.MODEL.checkpoint.ckp_flag:
        logger.info('===> Load ckp for <Mapping_net> from {}'.format(args.MODEL.checkpoint.ckp_path[0]))
        statedict     = torch.load(args.MODEL.checkpoint.ckp_path[0])
        Mapping_net.module.load_state_dict(statedict)
    statedict_encoder = torch.load(args.MODEL.checkpoint.ckp_path[1])
    Encoder_net.module.load_state_dict(statedict_encoder)
    statedict_attgan  = torch.load(args.MODEL.checkpoint.ckp_path[2])
    AttGAN_net.module.load_state_dict(statedict_attgan)
    

    logger.info('=> Set loss')
    L1_loss = nn.L1Loss()
    if args.USE_GPU:
        L1_loss.to(device)

    
    logger.info('=> Set optimizer')
    optimizer_mapping = torch.optim.Adam(Mapping_net.module.parameters(), \
                        lr=args.OPTIMIZER.lr_mapping, \
                        betas=(args.OPTIMIZER.Adam.beta1, args.OPTIMIZER.Adam.beta2), \
                        weight_decay=1e-4)
        
    """ Intrisic function"""
    def save_checkpoint(net, name, dir_path, epoch, bs, gap=1):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if epoch % gap == 0:
            path = os.path.join(dir_path, '{}_Batchsize_{}_Epoch_{}.pth'.format(name, bs, epoch))
            torch.save(net.module.state_dict(), path)
            logger.info('===> Save checkpoint at {}'.format(path))

    def adjust_learning_rate(args, epoch, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        lr = args.lr_mapping / (args.decay ** (epoch // args.decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def cor_square_error(x,y,eps=1e-12):
        return (1.0 - cosine_similarity(x,y,eps=eps)).mean()

    logger.info('=> Begin to train the model')
    Mapping_net.train()
    Encoder_net.eval()
    AttGAN_net.eval()
    G_net.model.eval()
    Tensor = torch.cuda.FloatTensor if args.USE_GPU else torch.FloatTensor
    
    for epoch in range(0, args.TRAIN.epochs):
        bar       = tqdm.tqdm(train_loader)
        loss_list = []
        cos_list  = []
        l1_list   = []
        for idx, (father, mother, child, childlabelr) in enumerate(bar):
            father = father.cuda()
            mother = mother.cuda()
            #child  = child[0].cuda()
            child_code = []
            childlabel = []
            #childlabel = childlabel.cuda()

            with torch.no_grad():
                father_code    = AttGAN_net(father, mode='enc')[-1]  #tensor of [bs,1024,4,4]
                mother_code    = AttGAN_net(mother, mode='enc')[-1]
                child_codetemp, _  = Encoder_net(child[0].cuda())
                child_code.append(child_codetemp)
                if len(child)==1:
                    child_code.append(child_codetemp)
                    child_code.append(child_codetemp)
                    child_code.append(child_codetemp)
                    childlabel.append(childlabelr[0].cuda())
                    childlabel.append(childlabelr[0].cuda())
                    childlabel.append(childlabelr[0].cuda())
                    childlabel.append(childlabelr[0].cuda())
                elif len(child)==2:
                    child_codetemp2, _  = Encoder_net(child[1].cuda())
                    child_code.append(child_codetemp2)
                    child_code.append(child_codetemp)
                    child_code.append(child_codetemp)
                    childlabel.append(childlabelr[0].cuda())
                    childlabel.append(childlabelr[1].cuda())
                    childlabel.append(childlabelr[0].cuda())
                    childlabel.append(childlabelr[0].cuda())
                    

            #train encoder network
            optimizer_mapping.zero_grad()
            recon_child_code   = Mapping_net(father_code.detach(), mother_code.detach())
            #print(len(recon_child_code))
            #print(recon_child_code[0].shape)
            #calculate losses
            l1loss = 0
            cosloss = 0
            #loss = 0
            decrease_ratio_list = [1.0, 1.0, 0.6, 0.3]    #显式指定
            compare_l1loss_list = []
            this_l1loss_list = []
            this_cosloss_list = []
            assert len(recon_child_code) == args.MODEL.mapping.out_num
            for i in range(args.MODEL.mapping.out_num):
                decrease_ratio = 1 - i / args.MODEL.mapping.out_num  #1.0 0.8 0.6 0.4 0.2
                this_l1loss_list.append(L1_loss(recon_child_code[i], child_code[i].detach()))
                this_cosloss_list.append(cor_square_error(recon_child_code[i], child_code[i].detach()))
                #decrease_ratio_list.append(decrease_ratio)   #显式指定时，这里要注释
                compare_l1loss_list.append(this_l1loss_list[i].item())

            
            min_index = compare_l1loss_list.index(min(compare_l1loss_list))
            decrease_ratio_list[min_index] = decrease_ratio_list[min_index] + 1.0
            
            for i in range(args.MODEL.mapping.out_num):
                l1loss += decrease_ratio_list[i]*this_l1loss_list[i]
                cosloss += decrease_ratio_list[i] * this_cosloss_list[i]
            loss = args.TRAIN.l1_weight * l1loss + args.TRAIN.cos_weight * cosloss
           # min_l1loss = L1_loss(recon_child_code[:, (args.MODEL.mapping.out_channels * min_index):(args.MODEL.mapping.out_channels * (min_index + 1))], child_code.detach())
           # loss    = args.TRAIN.l1_weight * l1loss + args.TRAIN.cos_weight * cosloss# + min_l1loss

            #backward
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer_mapping.step()
            #append losses into the corresponding list
            loss_list.append(loss.item())
            l1_list.append(l1loss.item())
            cos_list.append(cosloss.item())
            #tensorboard visualization
            if args.TRAIN.inference and idx%args.TRAIN.save_iter==0:
                uniform_code = torch.Tensor(father.shape[0], args.TRAIN.uniform_dim).uniform_().type_as(father)
                uniform_code = uniform_code.cuda()
                with torch.no_grad():       
                    recon_code = torch.cat((recon_child_code[min_index], childlabel[min_index], uniform_code), dim=1)
                    encode_code = torch.cat((child_code[1 if min_index==1 else 0], childlabel[1 if min_index==1 else 0], uniform_code), dim=1)
       
                    recon_img  = G_net.synthesize(recon_code)
                    encode_img = G_net.synthesize(encode_code.detach())

                vis_logger.visual_image(
                    {'gt_child': child[0],
                     'recon_child': recon_img,
                     'encode_img': encode_img},
                    (idx+1), normalize=True 
                )

            if idx%args.TRAIN.save_iter == 0:
                vis_logger.visual_text(
                    {'L1_loss_iter': l1loss.item(),
                     'cos_loss_iter': cosloss.item(),
                     'loss_iter': loss.item()},
                    idx,
                    main_tag='loss'
                )

        vis_logger.visual_text(
            {'L1_loss_epoch': np.mean(l1_list),
             'cos_loss_epoch': np.mean(cos_list),
             'loss_epoch': np.mean(loss_list)},
            epoch,
            main_tag='loss'
        )

        logger.info('[Epoch: {0}/{1}] [Learning Rate: {2:.7f}] | [L1_loss: {3:.5f}] [cos_loss: {4:.5f}] [loss: {5:.5f}] '.format(
            epoch,
            args.TRAIN.epochs,
            optimizer_mapping.param_groups[0]['lr'],
            np.mean(l1_list),
            np.mean(cos_list),
            np.mean(loss_list)
        ))

        #adjust the learning rate
        adjust_learning_rate(args.OPTIMIZER, epoch, optimizer_mapping)
        #save the checkpoints
        save_checkpoint(Mapping_net, 'MappingNet', args.MODEL.checkpoint.save_path[0], epoch, args.TRAIN.batch_size, gap=2)
        
