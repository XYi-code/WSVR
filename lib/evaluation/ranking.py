
from collections import defaultdict
import numpy as np
from ..utils.config import config, update_config
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from ..utils import to_numpy
from collections import defaultdict


def _binary_average_precision(y_true, y_score, sample_weight=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
    return auc(recall, precision)

def average_precision_score_modified(y_true, y_score, average="macro", sample_weight=None):
    if average == "macro":
        return _binary_average_precision(y_true, y_score, sample_weight)
    else:
        raise NotImplementedError("only macro")

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def map_cmc(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, topk=100, exclude_inf=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape

    if config.DATASET.NAMES != 'VehicleID':
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        ret = np.zeros(topk)
        aps = [] 
        num_valid_queries = 0
        top10_indices = []
        top10_labels = []
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            if exclude_inf:
                valid &= (distmat[i, indices[i]] < 1e11)
            if not np.any(matches[i, valid]): continue

            query_top10_indices = indices[i, :10]
            top10_indices.append(query_top10_indices)
            top10_labels.append(gallery_ids[query_top10_indices])
            
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score_modified(y_true, y_score))

            cmc_valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            if exclude_inf:
                cmc_valid &= (distmat[i, indices[i]] < 1e11)
            index = np.nonzero(matches[i, cmc_valid])[0]
            for j, k in enumerate(index):
                if k >= topk: break
                ret[k] += 1
                break
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps), ret.cumsum() / num_valid_queries, top10_indices, top10_labels
    else:
        
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        topk = min(100, n) 
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        
        
        ret = np.zeros(topk)
        aps = []
        num_valid_queries = 0
        for i in range(m):
           
            valid = (gallery_ids[indices[i]] != query_ids[i]) | (indices[i] != i)
            
            if not np.any(valid):
                continue

    
            query_top10_indices = indices[i, :10]
            top10_indices.append(query_top10_indices)
            top10_labels.append(gallery_ids[query_top10_indices])

            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): 
                continue
            aps.append(average_precision_score_modified(y_true, y_score))

            index = np.nonzero(matches[i, valid])[0]
            for j, k in enumerate(index):
                if k >= topk: break
                ret[k] += 1
                break
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps), ret.cumsum() / num_valid_queries


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    if config.DATASET.NAMES != 'VehicleID':
        distmat = to_numpy(distmat)
        m, n = distmat.shape
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):

            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:

                valid &= (gallery_cams[indices[i]] != query_cams[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                   
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps), ret.cumsum() / num_valid_queries
    else:
        distmat = to_numpy(distmat)
        m, n = distmat.shape
       
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        
        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):
            
            valid = (gallery_ids[indices[i]] != query_ids[i])
            
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
               
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return ret.cumsum() / num_valid_queries



def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    if config.DATASET.NAMES != 'VehicleID':
        distmat = to_numpy(distmat)
        m, n = distmat.shape

        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        aps = []
        for i in range(m):

            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score_modified(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)
    else:
        distmat = to_numpy(distmat)
        m, n = distmat.shape

        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        aps = []
        for i in range(m):

            valid = (gallery_ids[indices[i]] != query_ids[i])
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score_modified(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)




def analyze_distance_distribution(distmat, query_ids, gallery_ids, query_cams, gallery_cams):

    positive_distances = []
    negative_distances = []
    
    for i in range(distmat.shape[0]):
        for j in range(distmat.shape[1]):

            if query_cams[i] != gallery_cams[j]:

                if query_ids[i] == gallery_ids[j]: 
                    positive_distances.append(distmat[i, j])
                else:
                    negative_distances.append(distmat[i, j])
    

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.hist(positive_distances, bins=50, alpha=0.7, label='Positives')
    plt.axvline(x=0.1, color='r', linestyle='--', label='Threshold: 0.1')
    plt.title('Positive Pair Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(122)
    plt.hist(negative_distances, bins=50, alpha=0.7, label='Negatives')
    plt.axvline(x=0.1, color='r', linestyle='--', label='Threshold: 0.1')
    plt.title('Negative Pair Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('distance_analysis.png')
    plt.close() 
    
    return positive_distances, negative_distances

def normalize_distances(distmat):

    dist_min = np.min(distmat)
    dist_max = np.max(distmat)
    
    if dist_min < 0 or dist_max > 2:
        normalized = (distmat - dist_min) / (dist_max - dist_min)
        return normalized
    
    return distmat

def preprocess_distmat(distmat):


    dist_min = np.min(distmat)
    dist_max = np.max(distmat)
    normalized_dist = (distmat - dist_min) / (dist_max - dist_min)
    

    enhanced_dist = normalized_dist ** 2 

    flat_dist = enhanced_dist.flatten()
    hist, bins = np.histogram(flat_dist, 1000, density=True)
    cdf = hist.cumsum() / hist.sum()
    equalized_flat = np.interp(flat_dist, bins[:-1], cdf)
    equalized_dist = equalized_flat.reshape(distmat.shape)
    
    return equalized_dist



def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    if config.DATASET.NAMES != 'VehicleID':
        distmat = to_numpy(distmat)
        m, n = distmat.shape

        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):

            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:

                valid &= (gallery_cams[indices[i]] != query_cams[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
 
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return ret.cumsum() / num_valid_queries
    else:

        distmat = to_numpy(distmat)
        m, n = distmat.shape

        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
      
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
     
        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):
           
            valid = (gallery_ids[indices[i]] != query_ids[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                  
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return ret.cumsum() / num_valid_queries

def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):

    if config.DATASET.NAMES != 'VehicleID':
        distmat = to_numpy(distmat)
        m, n = distmat.shape
  
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)
   
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)
     
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
       
        aps = []
        for i in range(m):
        
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score_modified(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)
    else:
        distmat = to_numpy(distmat)
        m, n = distmat.shape
  
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
   
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
 
        aps = []
        for i in range(m):
     
            valid = (gallery_ids[indices[i]] != query_ids[i])
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score_modified(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)
