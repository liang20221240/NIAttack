import torch as th
from torch.nn.functional import normalize
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoraFull

def load_data(dataset="cora"):
    assert dataset in ["cora", "citeseer", "pubmed"]

    if dataset == "cora":
        dataset_obj = CoraGraphDataset()
    elif dataset == "citeseer":
        dataset_obj = CiteseerGraphDataset()
    else:
        dataset_obj = PubmedGraphDataset()

    g = dataset_obj[0]
    data = Data()
    data.features = th.FloatTensor(g.ndata["feat"])
    data.labels = th.LongTensor(g.ndata["label"])
    data.size = g.num_nodes()
    data.num_labels = dataset_obj.num_classes
    data.nodes = g.nodes().tolist()
    data.degree = g.out_degrees().tolist()
    data.graph = g
    data.g = g
    data.adj = build_dense_adj_from_edges(g)
    data.Prob = normalize(data.adj, p=1, dim=1)

    print("============ Successfully Load %s ===============" % dataset)

    return data


def split_data(data, NumTrain, NumTest, NumVal):
    idx_test = np.random.choice(data.size, NumTest, replace=False)
    without_test = np.array([i for i in range(data.size) if i not in idx_test])
    idx_train = without_test[np.random.choice(len(without_test),
                                              NumTrain,
                                              replace=False)]
    idx_val = np.array([
        i for i in range(data.size) if i not in idx_test if i not in idx_train
    ])
    idx_val = idx_val[np.random.choice(len(idx_val), NumVal, replace=False)]
    return idx_train, idx_val, idx_test


def load_CoraFull(dataset = "CoraFull"):

    data = CoraFull()
    g = data[0]

    data.features = g.ndata['feat']
    data.labels = g.ndata['label']
    data.size = 19793
    data.num_labels = 70
    data.nodes = g.nodes()
    data.degree = g.out_degrees()
    data.nodes = g.nodes().tolist()
    data.degree = g.out_degrees().tolist()

    data.graph = g

    # g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    data.g = g
    data.adj = g.adjacency_matrix(transpose=None).to_dense()
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1)

    print("============Successfully Load %s===============" % dataset)

    return data

def zero_gradients(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()
