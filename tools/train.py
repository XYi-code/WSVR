import argparse
import os
import os.path as osp
import sys
from collections import OrderedDict
import pprint
import random
import pdb
import time
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import numpy as np
import time
from tensorboardX import SummaryWriter
import _init_paths
from lib.datasets.dataset import DataSet
from lib import models
from lib.trainer import Trainer
from lib.evaluator import Evaluator
from lib.utils.data import transforms as T
from lib.utils.data.preprocessor import Preprocessor, UnsupervisedPreprocessor
from lib.utils.logging import Logger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.config import config, update_config
from lib.utils.netutils import get_optimizer

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--experiments', dest='cfg_file',
                        help='optional config file',
                        default='', type=str)
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--tb', type=str, default=True, help='observing optimisation process via tensorboard')
    parser.add_argument('--mlp', type=str, default='SMLC', help='multi label prediction methods')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix to add at the end of the log file')
    parser.add_argument('--use_dram', type=bool, default=False, help='Use DRAM for large-scale dataset')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(args.cfg_file)

    if args.mlp:
        config.MLP.TYPE = args.mlp

    if args.workers:
        config.WORKERS = args.workers
    print('Using config:')
    pprint.pprint(config)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writter = None
    if args.tb:
       
        _time_rightnow = time.strftime('%y%m%d_%H-%M-%S') 
        
      
        safe_suffix = args.suffix.replace("'", "").replace('"', "") if args.suffix else "default"
        
        log_dic = './logs/' + _time_rightnow + '_' + safe_suffix + '_'+ str(config.WSVR.LAMBDA_1) + '_' + str(config.WSVR.LAMBDA_2)
        if not os.path.exists(log_dic):
            os.makedirs(log_dic)
            writter = SummaryWriter(log_dic)
   

    sys.stdout = Logger(osp.join(config.OUTPUT_DIR, 'log.txt'))

    print("Dataset name from config:", config.DATASET.NAMES)  
    
 
    if config.DATASET.NAMES == 'VehicleID':
        try:
            from vehicleid import VehicleID
            dataset = VehicleID(root=config.DATASET.ROOT_DIR)
        except ImportError:
            raise ImportError("Error: Cannot find 'vehicleid.py' in your path. Please ensure it is in the root directory if you want to use VehicleID dataset.")
    else:
        dataset = DataSet(config.DATASET.ROOT, config.DATASET.DATASET)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(*config.MODEL.IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=config.DATASET.RE),
    ])
    test_transformer = T.Compose([
        T.Resize(config.MODEL.IMAGE_SIZE, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])


    if config.DATASET.NAMES == 'VehicleID':
        train_loader = DataLoader(
            UnsupervisedPreprocessor(dataset.train,
                        root=config.DATASET.ROOT, transform=train_transformer),
            batch_size=config.TRAIN.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=config.TRAIN.SHUFFLE, pin_memory=True)

        query_loader = DataLoader(
            Preprocessor(dataset.query,
                        root=config.DATASET.ROOT, transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        gallery_loader = DataLoader(
            Preprocessor(dataset.gallery,
                        root=config.DATASET.ROOT, transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)
    else:
        train_loader = DataLoader(
            UnsupervisedPreprocessor(dataset.train,
                        root=osp.join(dataset.images_dir, dataset.train_path), transform=train_transformer),
            batch_size=config.TRAIN.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=config.TRAIN.SHUFFLE, pin_memory=True)

        query_loader = DataLoader(
            Preprocessor(dataset.query,
                        root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        gallery_loader = DataLoader(
            Preprocessor(dataset.gallery,
                        root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

    if config.DATASET.DATASET == 'veri-wild':
        small_query_loader = DataLoader(
            Preprocessor(dataset.small_query,
                         root=osp.join(dataset.images_dir, dataset.small_query_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        small_gallery_loader = DataLoader(
            Preprocessor(dataset.small_gallery,
                         root=osp.join(dataset.images_dir, dataset.small_gallery_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)
        
        middle_query_loader = DataLoader(
            Preprocessor(dataset.middle_query,
                         root=osp.join(dataset.images_dir, dataset.middle_query_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        middle_gallery_loader = DataLoader(
            Preprocessor(dataset.middle_gallery,
                         root=osp.join(dataset.images_dir, dataset.middle_gallery_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

    model = models.create(config.MODEL.NAME, pretrained=config.MODEL.PRETRAINED)

    num_tgt = len(dataset.train)
    num_cam = int(config.DATASET.CAM_NUM)
    memory = models.create('memory', config.MODEL.FEATURES, num_tgt, num_cam)

    memory_loaded = False
    if config.TRAIN.RESUME:
        checkpoint = load_checkpoint(config.TRAIN.CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if 'state_dict_memory' in checkpoint:
            memory.load_state_dict(checkpoint['state_dict_memory'])
            memory_loaded = True
        config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
        print("=> Resuming from epoch {}{}".format(
            checkpoint['epoch'], ' (memory restored)' if memory_loaded else ' (memory will be reinitialized)'
        ))

    model = model.to(device)
    memory = memory.to(device)

    base_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        if 'base' in name:
            base_params.append(param)
        else:
            new_params.append(param)
    
    param_groups = [
        {'params': base_params, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}
    ]

    optimizer = get_optimizer(config, param_groups)

    trainer = Trainer(config, model, memory)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def adjust_lr(epoch):
        step_size = config.TRAIN.LR_STEP
        lr = config.TRAIN.LR * (config.TRAIN.LR_FACTOR ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    best_r1 = 0.0
    MEM_INIT = not memory_loaded

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_lr(epoch)
        current_lr = get_lr(optimizer)
        print('learning rate is %f' % (current_lr))

        if writter:
            writter.add_scalar("Learning_rate/train", current_lr, epoch)

        trainer.train(epoch, train_loader, optimizer, writter, gi=MEM_INIT)
        
        MEM_INIT = False 

        if epoch > 14:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'state_dict_memory': memory.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(config.OUTPUT_DIR, 'checkpoint_%d.pth.tar' % (epoch)))

        if epoch > 5:
            print('Test with latest model:')
            if len(query_loader.dataset) == 0:
                print("Query loader is empty")
            if len(gallery_loader.dataset) == 0:
                print("Gallery loader is empty")

            evaluator = Evaluator(model)
            r1, top10_indices, top10_labels = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, writter=writter, epoch=epoch, output_feature=config.TEST.OUTPUT_FEATURES)

            if config.DATASET.DATASET == 'veri-wild':
                evaluator.evaluate(middle_query_loader, middle_gallery_loader, dataset.middle_query, dataset.middle_gallery, writter=writter, epoch=epoch, output_feature=config.TEST.OUTPUT_FEATURES, suffix='middle')
                evaluator.evaluate(small_query_loader, small_gallery_loader, dataset.small_query, dataset.small_gallery, writter=writter, epoch=epoch, output_feature=config.TEST.OUTPUT_FEATURES, suffix='small')

            if r1 > best_r1:
                best_r1 = r1
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'state_dict_memory': memory.state_dict(),
                    'epoch': epoch + 1,
                }, fpath=osp.join(config.OUTPUT_DIR, 'best_checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.format(epoch))
            torch.cuda.empty_cache()
        
    print('Test with best model:')
    evaluator = Evaluator(model)
    checkpoint = load_checkpoint(osp.join(config.OUTPUT_DIR, 'best_checkpoint.pth.tar'))
    print('best model at epoch: {}'.format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, writter=None, epoch=None, output_feature=config.TEST.OUTPUT_FEATURES)
    if writter:
        writter.close()

if __name__ == '__main__':
    main()
