from collections import OrderedDict
import time
import os
import torch
import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
from .evaluation import cmc, mean_ap, map_cmc
from .utils.meters import AverageMeter
from .utils import to_torch
from .utils.config import config, update_config
from .utils.model_profile import print_model_summary, format_model_summary
import matplotlib.patches as patches


TRANSITION_MATRIX = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.218309859, 0, 0.78169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.479592, 0, 0.520408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.996269, 0, 0.003731343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.809524, 0, 0.190476, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.632911, 0, 0.367089, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.964744, 0, 0.035256, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04127, 0, 0.95873, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.053371, 0, 0.946629, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.967123, 0, 0.032877],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
])


CAMERAS = [6, 4, 7, 3, 8, 2, 9, 5, 10, 1, 0, 12, 11, 17, 16, 18, 19, 14, 13, 15]

CAMERA_TO_INDEX = {cam_id: idx for idx, cam_id in enumerate(CAMERAS)}

def get_camera_direction(source_cam):
     
    if isinstance(source_cam, (torch.Tensor, np.number)):
        source_cam = int(source_cam)
    
    if source_cam not in CAMERA_TO_INDEX:
      
        valid_cams = list(CAMERA_TO_INDEX.keys())
        return []
    
    source_idx = CAMERA_TO_INDEX[source_cam]
    transition_probs = TRANSITION_MATRIX[source_idx]
    
    non_zero_indices = np.nonzero(transition_probs)[0]
    

    max_idx = non_zero_indices[0]
    max_prob = transition_probs[max_idx]
    
    for idx in non_zero_indices[1:]:
        if transition_probs[idx] > max_prob:
            max_idx = idx
            max_prob = transition_probs[idx]
    

    target_cam = CAMERAS[max_idx]
    
    path = [target_cam]  
    
    next_target_idx = np.argmax(TRANSITION_MATRIX[max_idx])
    if TRANSITION_MATRIX[max_idx][next_target_idx] > 0:
        secondary_cam = CAMERAS[next_target_idx]
        if secondary_cam != target_cam:  
            path.append(secondary_cam)
    
    return path

def extract_cnn_feature(model, inputs, output_feature=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        inputs = to_torch(inputs)
        inputs = inputs.to(device)
        outputs = model(inputs, output_feature)
        outputs = outputs
    return outputs
def extract_features(model, data_loader, print_freq=1, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    num_images = 0

    features = OrderedDict()
    labels = OrderedDict()
    cameras = OrderedDict()  

    end = time.time()
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    for i, data in enumerate(data_loader):
       
        if len(data) == 4: 
            imgs, fnames, pids, camids = data
        elif len(data) == 6:  
            imgs, fnames, camids, pids, _, _ = data
        else:
            raise ValueError(f"Unexpected data format with {len(data)} elements")
        
        data_time.update(time.time() - end)
        if use_cuda:
            torch.cuda.synchronize()
        infer_start = time.time()
        outputs = extract_cnn_feature(model, imgs, output_feature)
        if use_cuda:
            torch.cuda.synchronize()
        batch_time.update(time.time() - infer_start)
        num_images += len(fnames)
        
        camids_list = camids.tolist() if isinstance(camids, torch.Tensor) else camids
        
        for fname, output, pid, camid in zip(fnames, outputs, pids, camids_list):
            features[fname] = output
            labels[fname] = pid
            cameras[fname] = int(camid) 

        
        end = time.time()

    if num_images > 0:
        ms_per_image = batch_time.sum / num_images * 1000.0
        print('Feature extraction: {:.3f} ms/image over {} images'.format(ms_per_image, num_images))

    return features, labels, cameras 

def _collect_query_gallery_tensors(query_features, gallery_features, query, gallery,
                                   query_cameras, gallery_cameras):
    query_feats, query_cams, query_fnames = [], [], []
    for fname, _, _, _, _, _ in query:
        if fname in query_features:
            query_feats.append(query_features[fname].unsqueeze(0))
            query_cams.append(query_cameras[fname])
            query_fnames.append(fname)

    gallery_feats, gallery_cams, gallery_fnames = [], [], []
    for fname, _, _, _, _, _ in gallery:
        if fname in gallery_features:
            gallery_feats.append(gallery_features[fname].unsqueeze(0))
            gallery_cams.append(gallery_cameras[fname])
            gallery_fnames.append(fname)

    x = torch.cat(query_feats, 0) if query_feats else torch.empty(0)
    y = torch.cat(gallery_feats, 0) if gallery_feats else torch.empty(0)
    return x, y, query_cams, gallery_cams


def pairwise_distance_standard(query_features, gallery_features, query, gallery,
                               query_cameras, gallery_cameras):
    x, y, _, _ = _collect_query_gallery_tensors(
        query_features, gallery_features, query, gallery, query_cameras, gallery_cameras
    )
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty((len(x), len(y)), device=x.device, dtype=x.dtype)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.mm(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def pairwise_distance(query_features, gallery_features, query, gallery, query_cameras, gallery_cameras):
    x, y, query_cams, gallery_cams = _collect_query_gallery_tensors(
        query_features, gallery_features, query, gallery, query_cameras, gallery_cameras
    )

    distmat = torch.full(
        (len(query_cams), len(gallery_cams)),
        1e12, device=x.device, dtype=x.dtype
    )

    if x.numel() == 0 or y.numel() == 0:
        return distmat

    for i, (q_cam, q_feat) in enumerate(zip(query_cams, x)):
        target_cams = get_camera_direction(q_cam)

        if not target_cams:
            distmat[i] = torch.pow(y - q_feat, 2).sum(dim=1)
            continue

        for j, (g_cam, g_feat) in enumerate(zip(gallery_cams, y)):
            if g_cam in target_cams:
                distmat[i, j] = torch.pow(g_feat - q_feat, 2).sum()

    return distmat


def evaluate_all(distmat, query=None, gallery=None,
    query_ids=None, gallery_ids=None,
    query_cams=None, gallery_cams=None, wr=None, ep=None, suffix=None,
    cmc_topk=(1, 5, 10, 20)):
    
    if config.DATASET.NAMES != 'VehicleID':
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _, _, _, _ in query]
            gallery_ids = [pid for _, pid, _, _, _, _ in gallery]
            query_cams = [cam for _, _, cam, _, _, _ in query]
            gallery_cams = [cam for _, _, cam, _, _, _ in gallery]
            query_paths = [path for _, _, _, _, _, path in query]
            gallery_paths = [path for _, _, _, _, _, path in gallery]
        else:
            assert (query_ids is not None and gallery_ids is not None
                    and query_cams is not None and gallery_cams is not None)
        
        
        query_images = None
        gallery_images = None
        
        q_pids = np.array(query_ids)
        g_pids = np.array(gallery_ids)
        q_camids = np.array(query_cams)
        g_camids = np.array(gallery_cams)
        print("q_pids", q_pids)
        print("g_pids", g_pids)
        
        mAP, all_cmc, top10_indices, top10_labels = map_cmc(
            distmat, q_pids, g_pids, q_camids, g_camids, exclude_inf=True
        )
        
        try:
            if 'query_paths' in locals() and 'gallery_paths' in locals():
                num_samples = min(5, len(query_paths))
                sample_indices = np.random.choice(len(query_paths), num_samples, replace=False)
                
                query_images = []
                gallery_images = []
                
                for idx in sample_indices:
                    q_img = cv2.imread(query_paths[idx])
                    if q_img is not None:
                        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
                        query_images.append(q_img)
                
                gallery_samples = min(20, len(gallery_paths)) 
                gallery_indices = np.random.choice(len(gallery_paths), gallery_samples, replace=False)
                
                for idx in gallery_indices:
                    g_img = cv2.imread(gallery_paths[idx])
                    if g_img is not None:
                        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
                        gallery_images.append(g_img)
        except Exception as e:
            query_images = None
            gallery_images = None
        
        save_path_1 = '/output/Veri/V1'
        os.makedirs(save_path_1, exist_ok=True)
        
            
        print('CMC Scores (camera-filtered retrieval)')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, all_cmc[k - 1]))
            if wr is not None:
                suffix_str = f"-{suffix}" if suffix else ""
                wr.add_scalar(f"CMC/Top-{k}{suffix_str}", all_cmc[k - 1], ep)
        
        return all_cmc[0], top10_indices, top10_labels
    else:
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _, _, _, _ in query]
            gallery_ids = [pid for _, pid, _, _, _, _ in gallery]
        else:
            assert (query_ids is not None and gallery_ids is not None)
            
        
        query_cams = gallery_cams = None
        mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        print('Mean AP: {:4.1%}'.format(mAP))
        
        if wr is not None:
            suffix_str = f"-{suffix}" if suffix else ""
            wr.add_scalar(f"Eval/MeanAP{suffix_str}", mAP, ep)
            
        print('CMC Scores')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, all_cmc[k - 1]))
            if wr is not None:
                suffix_str = f"-{suffix}" if suffix else ""
                wr.add_scalar(f"CMC/Top-{k}{suffix_str}", all_cmc[k - 1], ep)
        
        return all_cmc[0]

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self._profile_printed = False

    def _maybe_print_model_profile(self, output_feature=None, batch_size=1):
        if self._profile_printed:
            return None
        input_size = tuple(config.MODEL.IMAGE_SIZE)
        feature_name = output_feature or config.TEST.OUTPUT_FEATURES
        device = next(self.model.parameters()).device
        summary = print_model_summary(
            self.model,
            input_size=input_size,
            output_feature=feature_name,
            device=device,
            batch_size=batch_size,
            repeat=100,
        )
        self._profile_printed = True
        return summary

    def evaluate(self, query_loader, gallery_loader, query, gallery, writter=None, epoch=None, output_feature=None, suffix=None, profile_model=True):
        if profile_model:
            summary = self._maybe_print_model_profile(
                output_feature=output_feature,
                batch_size=getattr(query_loader, 'batch_size', 1) or 1,
            )
            if summary is not None and writter is not None and epoch is not None:
                suffix_str = f"-{suffix}" if suffix else ""
                writter.add_scalar(f"Model/Params{suffix_str}", summary['total_params'], epoch)
                writter.add_scalar(f"Model/MFLOPs{suffix_str}", summary['mflops'], epoch)
                writter.add_scalar(f"Model/LatencyMs{suffix_str}", summary['latency_ms'], epoch)
                writter.add_scalar(f"Model/SizeMB{suffix_str}", summary['model_size_mb'], epoch)


        query_features, _, query_cameras = extract_features(self.model, query_loader, 1, output_feature)

        gallery_features, _, gallery_cameras = extract_features(self.model, gallery_loader, 1, output_feature)
        
        # distmat_standard = pairwise_distance_standard(
        #     query_features,
        #     gallery_features,
        #     query,
        #     gallery,
        #     query_cameras,
        #     gallery_cameras,
        # )

        distmat = pairwise_distance(
            query_features,
            gallery_features,
            query,
            gallery,
            query_cameras,
            gallery_cameras,
        )

        # query_ids = [pid for _, pid, _, _, _, _ in query]
        # gallery_ids = [pid for _, pid, _, _, _, _ in gallery]
        # query_cams = [cam for _, _, cam, _, _, _ in query]
        # gallery_cams = [cam for _, _, cam, _, _, _ in gallery]
        # mAP_std, cmc_std, _, _ = map_cmc(
        #     distmat_standard,
        #     np.array(query_ids),
        #     np.array(gallery_ids),
        #     np.array(query_cams),
        #     np.array(gallery_cams),
        # )
        # for k in (1, 5, 10, 20):
        #     print('  top-{:<4}{:12.1%}'.format(k, cmc_std[k - 1]))

        return evaluate_all(distmat, query=query, gallery=gallery, wr=writter, ep=epoch, suffix=suffix)

def show_top_matches(query_paths, gallery_paths, top10_indices, top10_labels, query_labels, save_path):
    for idx, query_path in enumerate(query_paths):
        if idx % 10 != 0:  
            continue
            
        timestamp = int(time.time())
        query_img = cv2.imread(query_path)
        if query_img is None:
            print(f"Warning: could not load image {query_path}")
            continue

        if idx >= len(top10_indices):
            print(f"Warning: idx {idx} exceeds top10_indices length {len(top10_indices)}")
            continue
            
      
        if top10_indices[idx] is None or len(top10_indices[idx]) == 0:
            print(f"Warning: no valid top matches for query {idx}")
            continue
            
        plt.figure(figsize=(15, 5))
        ax = plt.subplot(1, 11, 1)
        if query_img is not None:
            ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        ax.set_title("Query")
        ax.axis('off')
        
        for i, match_idx in enumerate(top10_indices[idx][:10]):  
            if match_idx >= len(gallery_paths):
                print(f"Warning: match_idx {match_idx} exceeds gallery_paths length {len(gallery_paths)}")
                continue
                
            match_path = gallery_paths[match_idx]
            img = cv2.imread(match_path)
            if img is None:
                print(f"Warning: could not load image {match_path}")
                continue
                
            ax = plt.subplot(1, 11, i+2)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')

        
            if idx < len(query_labels) and i < len(top10_labels[idx]):
                if top10_labels[idx][i] == query_labels[idx]:
                    edge_color = 'g'
                else:
                    edge_color = 'r'
                rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=2, edgecolor=edge_color, facecolor='none')
                ax.add_patch(rect)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'reid_result_{timestamp}_{idx}.png'), format='png', dpi=300)
        plt.close()
