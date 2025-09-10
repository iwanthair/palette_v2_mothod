import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
from data.dataset import FloorPlanDatasetTrajOnly

# TRAIN_DIR_Q0 = 'Dataset_Scale100_SEPE/train/Condition_1'
# TRAIN_DIR_Q1 = 'Dataset_Scale100_SEPE/train/Condition_2'
# TRAIN_DIR_TARGET = 'Dataset_Scale100_SEPE/train/Target'
# VAL_DIR_Q0 = 'Dataset_Scale100_SEPE/test/Condition_1'
# VAL_DIR_Q1 = 'Dataset_Scale100_SEPE/test/Condition_2'
# VAL_DIR_TARGET = 'Dataset_Scale100_SEPE/test/Target'

# TRAIN_DIR_Q0 = 'Dataset_Scale100_SEPE/Selected_50_train/Condition_1'
# TRAIN_DIR_Q1 = 'Dataset_Scale100_SEPE/Selected_50_train/Condition_2'
# TRAIN_DIR_TARGET = 'Dataset_Scale100_SEPE/Selected_50_train/Target'
# VAL_DIR_Q0 = 'Dataset_Scale100_SEPE/Selected_50_test/Condition_1'
# VAL_DIR_Q1 = 'Dataset_Scale100_SEPE/Selected_50_test/Condition_2'
# VAL_DIR_TARGET = 'Dataset_Scale100_SEPE/Selected_50_test/Target'

# TRAIN_DIR_Q0 = 'Dataset_Scale100_SExPE/train/Condition_1'
# TRAIN_DIR_Q1 = 'Dataset_Scale100_SExPE/train/Condition_2'
# TRAIN_DIR_TARGET = 'Dataset_Scale100_SExPE/train/Target'
# VAL_DIR_Q0 = 'Dataset_Scale100_SExPE/test/Condition_1'
# VAL_DIR_Q1 = 'Dataset_Scale100_SExPE/test/Condition_2'
# VAL_DIR_TARGET = 'Dataset_Scale100_SExPE/test/Target'

TRAIN_DIR_Q0 = 'Dataset_Scale100_SExPE/Selected_50_train/Condition_1'
TRAIN_DIR_Q1 = 'Dataset_Scale100_SExPE/Selected_50_train/Condition_2'
TRAIN_DIR_TARGET = 'Dataset_Scale100_SExPE/Selected_50_train/Target'
VAL_DIR_Q0 = 'Dataset_Scale100_SExPE/Selected_50_test/Condition_1'
VAL_DIR_Q1 = 'Dataset_Scale100_SExPE/Selected_50_test/Condition_2'
VAL_DIR_TARGET = 'Dataset_Scale100_SExPE/Selected_50_test/Target'

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    ''' set dataloader '''
    train_dataset = FloorPlanDatasetTrajOnly(TRAIN_DIR_Q1, TRAIN_DIR_TARGET, image_size=[128, 128])
    print("Number of training samples:", len(train_dataset))
    dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print("Number of training batches:", len(dataloader_train))
    print("Shape of dataloader:", next(iter(dataloader_train))['cond_image'].shape)

    val_dataset = FloorPlanDatasetTrajOnly(VAL_DIR_Q1, VAL_DIR_TARGET, image_size=[128, 128])
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    print("Number of validation samples:", len(val_dataset))

    # phase_loader, val_loader = dataloader_val, dataloader_val # validation on testing set
    phase_loader, val_loader = dataloader_train, dataloader_train # validation on training set
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Palette_scalar100_abl_traj.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
