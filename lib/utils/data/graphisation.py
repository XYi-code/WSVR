import numpy as np
import scipy.sparse as sp
import torch

def one_hot_encoding_wit_digit(labels,num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def gen_egdes(features,pid,thr,epoch=None,is_save=True):
    unordered_edges=[]
    _shape = np.shape(features)
    for _p,_f1 in enumerate(zip(features,pid)):
        for _, _f2 in enumerate(zip(features[_p+1:],pid[_p+1:])):
            _sim = np.dot(_f1[0],_f2[0])
            if _sim > thr:
                unordered_edges.append([_f1[1],_f2[1]])
    unordered_edges = np.array(unordered_edges,dtype=np.int32)
    return unordered_edges 

def gen_egdes_faster(features,thr):

    unordered_edges=[]
    _shape = np.shape(features)
    _sim= np.matmul(features,features.T)
    _idx_low = np.triu_indices(_shape[0],1)
    _sim[_idx_low] = -1.0
    _idx = np.argwhere(_sim>thr)
    unordered_edges = np.array(_idx,dtype=np.int32)
    return unordered_edges 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices,values,shape)


def generate_graph_input(features,labels,pid,alpha=0.5,num_classes=20000):
    tmp_feature = features.detach().numpy()
    features = sp.csr_matrix(tmp_feature, dtype=np.float32)
    labels = one_hot_encoding_wit_digit(labels.detach().numpy(),num_classes=num_classes)


    pid = pid.detach().numpy()

    idx = np.array(pid, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = gen_egdes_faster(tmp_feature,thr=alpha)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    return adj, features, labels



def generate_subgraph_input(features,pids,alpha=0.0,num_classes=20000):
    tmp_feature = features.detach().cpu().numpy()
    features = sp.csr_matrix(tmp_feature, dtype=np.float32)
    pids = np.argsort(pids.detach().cpu().numpy())
    labels = one_hot_encoding_wit_digit(pids,num_classes=num_classes)

    idx = np.array(pids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = gen_egdes_faster(tmp_feature,thr=alpha)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels