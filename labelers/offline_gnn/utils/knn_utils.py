import os
import time
import torch
import numpy as np
import scipy.sparse as sp
import faiss as faiss_utils

class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name, time.time() - self.start), flush=True)
        return exc_type is None

class FaissKNN():
    def __init__(self,feats, dist_def, k, knn_ofn='', build_graph_method='gpu'):
        if knn_ofn != '' and os.path.exists(knn_ofn):
                print('[faiss] read knns from {}'.format(knn_ofn))
                self.knns = [(knn[0, :].astype(np.int32), knn[1, :].astype(np.float32))
                             for knn in np.load(knn_ofn)['knns']]
        else:
            if build_graph_method == 'gpu':
                res = faiss_utils.StandardGpuResources()
                with Timer('build index'):
                    feats = feats.astype('float32')
                    if dist_def == 'cosine:':
                        index = faiss_utils.IndexFlatIP(feats.shape[1])
                    else:
                        index = faiss_utils.IndexFlatL2(feats.shape[1])
                    # index.add(feats)
                    gpu_index_flat = faiss_utils.index_cpu_to_gpu(res, 0, index)
                    gpu_index_flat.add(feats)
                with Timer('query topk {}'.format(k)):
                    sims, ners = gpu_index_flat.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                 for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)

            if build_graph_method == 'approx':
                res = faiss_utils.StandardGpuResources()
                with Timer('build index'):
                    d = feats.shape[1]
                    feats = feats.astype('float32')
                    nlist = 256
                    m = 8
                    if dist_def == 'cosine:':
                        quantizer = faiss_utils.IndexFlatIP(d)
                    else:
                        quantizer = faiss_utils.IndexFlatL2(d)
                    index = faiss_utils.IndexIVFPQ(quantizer, d, nlist, m, 8)
                    gpu_index_flat = faiss_utils.index_cpu_to_gpu(res, 0, index)
                    assert not gpu_index_flat.is_trained
                    gpu_index_flat.train(feats)
                    assert gpu_index_flat.is_trained
                    gpu_index_flat.add(feats)
                with Timer('query topk {}'.format(k)):
                    sims, ners = gpu_index_flat.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                 for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)

            elif build_graph_method == 'cpu':
                with Timer('build index'):
                    feats = feats.astype('float32')
                    if dist_def == 'cosine:':
                        index = faiss_utils.IndexFlatIP(feats.shape[1])
                    else:
                        index = faiss_utils.IndexFlatL2(feats.shape[1])
                    index.add(feats)
                with Timer('query topk {}'.format(k)):
                    sims, ners = index.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                 for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)
    def get_knns(self):
        return self.knns


def global_build(feature_root, dist_def, k, save_filename, build_graph_method):
    if type(feature_root) is str:
        full_feat = np.load(feature_root)
    else:
        full_feat = feature_root
    knn_save_path = save_filename + '.npz'
    knn_graph = FaissKNN(full_feat, dist_def, k, knn_save_path, build_graph_method)
    return knn_graph

def fast_knns2spmat(knns, th_sim=0.2, edge_weight=True):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :]
    sims = knns[:, 1, :]
    assert -eps <= sims.min() <= sims.max() <= 1 + eps, "min: {}, max: {}".format(sims.min(), sims.max())
    # remove low similarity
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    if not edge_weight:
        data = np.ones(data.shape)
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sgc_precompute(features, knn_graph, self_weight=0, edge_weight=True, degree=1):
    features = torch.FloatTensor(features.astype(np.float32))
    knns = knn_graph.get_knns()
    adj = fast_knns2spmat(knns, th_sim=0, edge_weight=edge_weight)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize(adj)
    adj = adj + self_weight * sp.eye(adj.shape[0])
    adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    with Timer('sgc_precompute', True):
        for i in range(degree):
            features = torch.spmm(adj, features)
    return features.detach().numpy()

