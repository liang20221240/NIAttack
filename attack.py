import torch
import dgl
import random
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

def dgl_to_sparse_adj(g):
    src, dst = g.edges()
    n = g.num_nodes()

    indices = torch.stack([src, dst], dim=0)
    values = torch.ones(indices.size(1))

    adj = torch.sparse_coo_tensor(indices, values, (n, n))
    return adj.coalesce()

def normalize_adj(adj):
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv = 1.0 / torch.clamp(deg, min=1)

    row, col = adj.indices()
    values = adj.values() * deg_inv[row]

    P = torch.sparse_coo_tensor(adj.indices(), values, adj.size())
    return P.coalesce()

def sparse_power(P, L):
    result = P
    for _ in range(L - 1):
        result = torch.sparse.mm(result, P)
    return result

def compute_influence(P_L, labels):
    P_dense = P_L.to_dense()

    num_classes = labels.max().item() + 1
    Y = torch.nn.functional.one_hot(labels, num_classes).float()

    same_class = (P_dense @ Y) * Y
    same_class = same_class.sum(dim=1)

    I = 1.0 - same_class
    return I

def compute_degree(adj):
    return torch.sparse.sum(adj, dim=1).to_dense()

def compute_NL(adj, labels, nodes):
    row, col = adj.indices()

    NL = {}

    for u in nodes:
        mask = (row == u)
        neighbors = col[mask]

        if len(neighbors) == 0:
            NL[u.item()] = 0
        else:
            NL[u.item()] = len(torch.unique(labels[neighbors]))

    return NL

def NI_attack(adj, labels, P_L, r):
    n = labels.size(0)

    I = compute_influence(P_L, labels)
    degree = compute_degree(adj)

    S = set()

    while len(S) < r:

        mask = torch.ones(n, dtype=torch.bool)
        if len(S) > 0:
            mask[list(S)] = False

        # Step 1: S1 = argmax I(u)
        I_masked = I.clone()
        I_masked[~mask] = -1

        max_I = torch.max(I_masked)
        S1 = torch.where(I_masked == max_I)[0]

        if len(S1) == 1:
            S.add(S1.item())
            continue

        # Step 2: S2 = argmax D(u)
        deg_S1 = degree[S1]
        max_D = torch.max(deg_S1)
        S2 = S1[deg_S1 == max_D]

        if len(S2) == 1:
            S.add(S2.item())
            continue

        # Step 3: S3 = argmax NL(u)
        NL = compute_NL(adj, labels, S2)
        max_NL = max(NL.values())

        S3 = [u for u in S2.tolist() if NL[u] == max_NL]

        if len(S3) == 1:
            S.add(S3[0])
        else:
            s = random.choice(S3)
            S.add(s)

    return list(S)
