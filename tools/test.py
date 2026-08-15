import argparse
import os
import os.path as osp
import sys
import pprint

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import _init_paths
from lib.datasets.dataset import DataSet
from lib import models
from lib.evaluator import Evaluator
from lib.utils.data import transforms as T
from lib.utils.data.preprocessor import Preprocessor
from lib.utils.logging import Logger
from lib.utils.serialization import load_checkpoint
from lib.utils.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised ReID via Multi-label Classification')
    parser.add_argument('--experiments', dest='cfg_file',
                        help='optional config file',
                        default='experiments/veri776.yml', type=str)
    parser.add_argument('--dataset', type=str, default=None, help='force dataset name')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    return parser.parse_args()

def main():
    args = parse_args()
    update_config(args.cfg_file)
    if args.dataset:
        config.DATASET.NAMES = args.dataset
    elif 'veri' in args.cfg_file.lower():
        config.DATASET.NAMES = 'VeRi'
        
    if args.workers:
        config.WORKERS = args.workers
        
    print('=' * 40)
    print(f"🚀 [CONFIRMED] Dataset name applied to config: {config.DATASET.NAMES}")
    print('=' * 40)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.stdout = Logger(osp.join(config.OUTPUT_DIR, 'log-eval.txt'))

    if config.DATASET.NAMES == 'VehicleID':
        from vehicleid import VehicleID
        dataset = VehicleID(root=config.DATASET.ROOT_DIR)
    else:
        dataset = DataSet(config.DATASET.ROOT, config.DATASET.DATASET)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize(config.MODEL.IMAGE_SIZE, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    if config.DATASET.NAMES == 'VehicleID':
        query_loader = DataLoader(
            Preprocessor(dataset.query, root=config.DATASET.ROOT, transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor(dataset.gallery, root=config.DATASET.ROOT, transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
    else:
        query_loader = DataLoader(
            Preprocessor(dataset.query, root=osp.join(dataset.images_dir, dataset.query_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir, dataset.gallery_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)

    if config.DATASET.DATASET == 'veri-wild':
        small_query_loader = DataLoader(
            Preprocessor(dataset.small_query, root=osp.join(dataset.images_dir, dataset.small_query_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
        small_gallery_loader = DataLoader(
            Preprocessor(dataset.small_gallery, root=osp.join(dataset.images_dir, dataset.small_gallery_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
        middle_query_loader = DataLoader(
            Preprocessor(dataset.middle_query, root=osp.join(dataset.images_dir, dataset.middle_query_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)
        middle_gallery_loader = DataLoader(
            Preprocessor(dataset.middle_gallery, root=osp.join(dataset.images_dir, dataset.middle_gallery_path), transform=transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS, shuffle=False, pin_memory=True)

    model = models.create(config.MODEL.NAME)
    checkpoint = load_checkpoint(config.TEST.MODEL_FILE)
    print(f' Loaded checkpoint from epoch: {checkpoint["epoch"]}')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = nn.DataParallel(model).to(device) if device.type == 'cuda' else model.to(device)

    print('Start evaluation...')
    evaluator = Evaluator(model)
    
    if config.DATASET.DATASET == 'veri-wild':
        print('======= Large Set Eval =======')
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, writter=None, epoch=None, output_feature=config.TEST.OUTPUT_FEATURES)

    if config.DATASET.DATASET == 'veri-wild':
        print('======= Middle Set Eval =======')
        evaluator.evaluate(middle_query_loader, middle_gallery_loader, dataset.middle_query, dataset.middle_gallery, writter=None, epoch=None, output_feature=config.TEST.OUTPUT_FEATURES)
        print('======= Small Set Eval =======')
        evaluator.evaluate(small_query_loader, small_gallery_loader, dataset.small_query, dataset.small_gallery, writter=None, epoch=None, output_feature=config.TEST.OUTPUT_FEATURES)

if __name__ == '__main__':
    main()
