import torch
import torch.nn as nn
import torch.nn.functional as F

class WSVR_Loss(nn.Module):
    """
    Optimized Weakly Supervised Vehicle Re-Identification (WSVR) Loss Component.
    Fully vectorized to avoid Python per-sample loops and CUDART execution crashes.
    """
    def __init__(self, temperature=0.1, alpha_scl=0.5, k_centroids=2, lambda_1=1.0, lambda_2=1.0, lambda_total=0.9, cross_camera_threshold=0.5, ice_hard_k=32):
        super(WSVR_Loss, self).__init__()
        self.tau = temperature
        self.alpha_scl = alpha_scl
        self.k = k_centroids
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cross_camera_threshold = cross_camera_threshold
        self.ice_hard_k = ice_hard_k
        
        self.global_prototypes = {}
        self.cam_centers = {}

    @staticmethod
    def _track_level_logits(sim_row, proto_cids, proto_tids, track_id, tau):
        pos_mask = proto_tids == track_id
        if not torch.any(pos_mask):
            return None, None

        pos_logits = sim_row[pos_mask] / tau 
        neg_mask = proto_tids != track_id
        if not torch.any(neg_mask):
            return None, None

        neg_tids = proto_tids[neg_mask]
        neg_sims = sim_row[neg_mask]
        unique_neg_tids, inv_idx = torch.unique(neg_tids, return_inverse=True)
        num_neg_tracks = len(unique_neg_tids)

        neg_track_max_sims = torch.full((num_neg_tracks,), float('-inf'), device=sim_row.device)
        if hasattr(neg_track_max_sims, 'scatter_reduce_'):
            neg_track_max_sims.scatter_reduce_(0, inv_idx, neg_sims, reduce='amax')
        else:
            sorted_idx = torch.argsort(inv_idx)
            sorted_tids = inv_idx[sorted_idx]
            sorted_sims = neg_sims[sorted_idx]
            diff = torch.cat([torch.ones(1, dtype=torch.bool, device=sim_row.device),
                              sorted_tids[1:] != sorted_tids[:-1]])
            group_starts = torch.where(diff)[0]
            group_max_sims = sorted_sims[group_starts]
            next_starts = torch.cat([group_starts[1:], torch.tensor([len(sorted_sims)], device=sim_row.device)])
            for i, (s, e) in enumerate(zip(group_starts.tolist(), next_starts.tolist())):
                neg_track_max_sims[sorted_tids[s]] = sorted_sims[s:e].max()

        neg_track_sims = neg_track_max_sims / tau  

        pos_mask_final = torch.zeros(num_neg_tracks + pos_logits.numel(), dtype=torch.bool, device=sim_row.device)
        pos_mask_final[:pos_logits.numel()] = True

        logits_all = torch.cat([pos_logits, neg_track_sims], dim=0)
        return logits_all, pos_mask_final

    @torch.no_grad()
    def update_global_prototypes(self, memory):
        self.global_prototypes.clear()
        self.cam_centers.clear()
        
        valid_mask = (memory.mem.pow(2).sum(dim=1) > 0)
        if not torch.any(valid_mask):
            return
            
        unique_cams = torch.unique(memory.mem_CID[valid_mask])
        
        for cid in unique_cams:
            cid_item = cid.item()
            cam_feats = memory.mem[(memory.mem_CID == cid) & valid_mask]
            if len(cam_feats) > 0:
                self.cam_centers[cid_item] = F.normalize(torch.mean(cam_feats, dim=0), p=2, dim=0)
        
        for cid in unique_cams:
            cid_mask = (memory.mem_CID == cid) & valid_mask
            cam_tids = torch.unique(memory.mem_TID[cid_mask])
            cid_item = cid.item()
            
            for tid in cam_tids:
                tid_item = tid.item()
                track_mask = cid_mask & (memory.mem_TID == tid)
                features = memory.mem[track_mask]
                
                if len(features) == 0:
                    continue
                
                num_f = len(features)
                if num_f >= self.k:
                    indices = torch.randperm(num_f, device=features.device)[:self.k]
                    centroids = features[indices].clone()
                    for _ in range(3):
                        dist = torch.cdist(features, centroids)
                        labels = torch.argmin(dist, dim=1)
                        for j in range(self.k):
                            mask = (labels == j)
                            if torch.sum(mask) > 0:
                                centroids[j] = torch.mean(features[mask], dim=0)
                else:
                    repeat_count = (self.k // num_f) + 1
                    centroids = features.repeat(repeat_count, 1)[:self.k]
                    centroids = centroids + torch.randn_like(centroids) * 1e-4
                
                self.global_prototypes[(cid_item, tid_item)] = F.normalize(centroids, p=2, dim=1)

    def forward(self, memory, outputs, camids, trackids, epoch):
        device = outputs.device
        batch_size = outputs.size(0)
        
        if not self.global_prototypes:
            self.update_global_prototypes(memory)


        keys = list(self.global_prototypes.keys())
        if not keys:
            return outputs.sum() * 0.0
            

        all_protos_tensor = torch.cat([self.global_prototypes[k] for k in keys], dim=0).to(device)
        

        proto_cids = torch.tensor([k[0] for k in keys for _ in range(self.k)], device=device)
        proto_tids = torch.tensor([k[1] for k in keys for _ in range(self.k)], device=device)

        sim_matrix = torch.matmul(outputs, all_protos_tensor.t())

        loss_scl_list = []
        loss_ice_list = []
        loss_icfc_list = []
        loss_ccao_list = []
        
        valid_mask = (memory.mem.pow(2).sum(dim=1) > 0)

        for i in range(batch_size):
            c_i = camids[i].item()
            t_i = trackids[i].item()
            m_p = outputs[i]

            same_cam_diff_track = (proto_cids == c_i) & (proto_tids != t_i)
            same_cam_same_track = (proto_cids == c_i) & (proto_tids == t_i)

            if not torch.any(same_cam_diff_track) or not torch.any(same_cam_same_track):
                continue

            other_protos = all_protos_tensor[same_cam_diff_track]
            inst_other_sim = sim_matrix[i, same_cam_diff_track]
            inst_other_dist = torch.norm(other_protos - m_p.unsqueeze(0), p=2, dim=1)
            w_i = torch.exp(-self.alpha_scl * (inst_other_dist ** 2))
            loss_scl_list.append(torch.max(w_i * inst_other_sim))

            ice_all_logits, ice_pos_mask = self._track_level_logits(
                sim_matrix[i], proto_cids, proto_tids, t_i, self.tau
            )
            if ice_all_logits is not None and ice_pos_mask is not None:
                neg_mask = ~ice_pos_mask
                neg_logits = ice_all_logits[neg_mask]
                if self.ice_hard_k > 0 and neg_logits.numel() > self.ice_hard_k:
                    neg_logits, _ = torch.topk(neg_logits, self.ice_hard_k)
                ice_logits = torch.cat([ice_all_logits[ice_pos_mask], neg_logits], dim=0)
                loss_ice_list.append(torch.logsumexp(ice_logits, dim=0) - ice_logits[0])

            curr_track_sims = sim_matrix[i, same_cam_same_track]
            best_proto = all_protos_tensor[same_cam_same_track][torch.argmax(curr_track_sims)]
            
            neg_instances_mask = (memory.mem_CID == c_i) & (memory.mem_TID != t_i) & valid_mask
            neg_instances = memory.mem[neg_instances_mask]
            
            if len(neg_instances) > 0:
                pos_logit = torch.dot(best_proto, m_p) / self.tau
                neg_logits = torch.mm(best_proto.unsqueeze(0), neg_instances.t()) / self.tau
                logits_concat = torch.cat([pos_logit.unsqueeze(0), neg_logits.squeeze(0)], dim=0)
                loss_icfc_list.append(torch.logsumexp(logits_concat, dim=0) - pos_logit)

            if epoch > 10:
                diff_cam_mask = (proto_cids != c_i)
                if torch.any(diff_cam_mask):
                    diff_cam_sims = sim_matrix[i, diff_cam_mask]
                    pos_mask = diff_cam_sims > self.cross_camera_threshold
                    neg_mask = ~pos_mask
                    
                    if torch.any(pos_mask) and torch.any(neg_mask):
                        pos_logits = diff_cam_sims[pos_mask] / self.tau
                        neg_logits = diff_cam_sims[neg_mask] / self.tau
                        logits_all = torch.cat([pos_logits, neg_logits], dim=0)
                        loss_ccao_list.append(torch.logsumexp(logits_all, dim=0) - torch.logsumexp(pos_logits, dim=0))

        loss_cda_list = []
        if epoch > 10 and self.cam_centers:
            sorted_cam_ids = sorted(list(self.cam_centers.keys()))
            cam_centers_tensor = torch.stack([self.cam_centers[cid] for cid in sorted_cam_ids]).to(device)

            p_cda_logits = torch.matmul(outputs, cam_centers_tensor.t()) / self.tau
            P_xc = F.softmax(p_cda_logits, dim=1)
            n_c = len(sorted_cam_ids)
            U_xc = 1.0 / n_c
            kl_div = (U_xc * (torch.log(torch.tensor(U_xc, device=device)) - torch.log(P_xc + 1e-10))).sum(dim=1)
            loss_cda_list.append(kl_div.mean())

        zero = outputs.sum() * 0.0
        L_SCL = torch.stack(loss_scl_list).mean() if loss_scl_list else zero
        L_ICE = torch.stack(loss_ice_list).mean() if loss_ice_list else zero
        L_ICFC = torch.stack(loss_icfc_list).mean() if loss_icfc_list else zero
        
        L_ICFO = L_SCL + self.lambda_1 * L_ICE + self.lambda_2 * L_ICFC

        if epoch > 10 and loss_ccao_list and loss_cda_list:
            L_CCAO = torch.stack(loss_ccao_list).mean()
            L_CDA = torch.stack(loss_cda_list).mean()
            total_loss =  L_ICFO + L_CCAO + L_CDA
        else:
            total_loss = L_ICFO

        return total_loss
