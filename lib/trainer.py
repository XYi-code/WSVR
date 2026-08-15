import time
import numpy as np
import torch
from .utils.meters import AverageMeter
from .utils.plot_figures import utils_for_fig3
from .utils.mlp_statistics import precision_recall

from .loss import WSVR_Loss
import pdb
from .utils.config import config, update_config

class Trainer(object):
    def __init__(self, cfg, model, memory, use_dram=False):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.memory = memory
        self.use_dram = use_dram
        self.eval_mlp = True
        self.criterion = WSVR_Loss(
            temperature=cfg.WSVR.TEMP,
            alpha_scl=cfg.WSVR.ALPHA_SCL,
            k_centroids=cfg.WSVR.K_CENTERS,
            lambda_1=cfg.WSVR.LAMBDA_1, 
            lambda_2=cfg.WSVR.LAMBDA_2, 
            lambda_total=cfg.WSVR.LAMBDA_TOTAL,
            cross_camera_threshold=cfg.WSVR.T,
            ice_hard_k=getattr(cfg.WSVR, 'ICE_HARD_K', 32),
        ).to(self.device)


    def train(self, epoch, data_loader, optimizer, writer, gi=True, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()


        if gi:
            print('Memory Re-initisliation')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs, camid, tid, pids = self._parse_data(inputs)
                    outputs = self.model(inputs, 'l2feat')
                    self.memory.store(outputs, camid, tid, pids)
            print('Done!')

        if epoch % 5 == 0 and epoch != 0:
            print('Look-up table Overhaul - [reinitialising]')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs, camid, tid, pids = self._parse_data(inputs)
                    outputs = self.model(inputs, 'l2feat')
                    self.memory.store(outputs, camid, tid, pids)
                    if (i + 1) % print_freq == 0:
                        print('[Reinitilisaing] Overhaul (%.3f %%) is finished' % (i / len(data_loader) * 100.0))
                print('Dictionary overhaul is finished. - Overhaul (100%%) is finished')

        precision = 0.0
        recall = 0.0
        _num_positive = 0

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, camid, tid, pids = self._parse_data_v2(inputs)
            camid = camid.to(self.device)
            tid = tid.to(self.device)

            outputs = self.model(inputs, 'l2feat')
            
            loss = self.criterion(self.memory, outputs, camid, tid, epoch=epoch)
            losses.update(loss.item(), outputs.size(0))
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + i)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}) " \
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)
                print(log)
    
        with torch.no_grad():
            self.memory.set_cam_memory() 
            self.criterion.update_global_prototypes(self.memory) 
        torch.cuda.empty_cache()

    def _parse_data(self, inputs):
        imgs, _t1, camid, _, pids, tid = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        tid = tid.to(self.device)
        return inputs, camid, tid, pids

    def _parse_data_v2(self, inputs):
        imgs, _t1, camid, _, pids, tid = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        tid = tid.to(self.device)
        return inputs, camid, tid, pids
