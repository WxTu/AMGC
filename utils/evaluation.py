from sklearn import metrics
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, fts_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(fts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_fts):
        h1 = F.relu(self.fc1(input_fts))
        h2 = self.fc2(h1)
        return F.log_softmax(h2, dim=1)


def class_eva(train_fts, train_lbls, test_fts, test_lbls):
    test_featured_idx = np.where(test_fts.sum(1) != 0)[0]
    test_non_featured_idx = np.where(test_fts.sum(1) == 0)[0]

    featured_test_fts = test_fts[test_featured_idx]
    featured_test_lbls = test_lbls[test_featured_idx]
    non_featured_test_lbls = test_lbls[test_non_featured_idx]

    fts_dim = train_fts.shape[1]
    hid_dim = 64
    n_class = int(max(max(train_lbls), max(test_lbls)) + 1)
    is_cuda = torch.cuda.is_available()

    model = MLP(fts_dim, hid_dim, n_class)
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    featured_test_lbls_arr = featured_test_lbls.copy()

    train_fts = torch.from_numpy(train_fts).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    featured_test_fts = torch.from_numpy(featured_test_fts).float()
    featured_test_lbls = torch.from_numpy(featured_test_lbls).long()
    if is_cuda:
        train_fts = train_fts.cuda()
        train_lbls = train_lbls.cuda()
        featured_test_fts = featured_test_fts.cuda()
        featured_test_lbls = featured_test_lbls.cuda()

    acc_list = []
    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_fts)

        loss = F.nll_loss(outputs, train_lbls)
        loss.backward()
        optimizer.step()

        model.eval()
        featured_test_outputs = model(featured_test_fts)
        test_loss = F.nll_loss(featured_test_outputs, featured_test_lbls)
        if is_cuda:
            featured_test_outputs = featured_test_outputs.data.cpu().numpy()
        else:
            featured_test_outputs = featured_test_outputs.data.numpy()
        featured_preds = np.argmax(featured_test_outputs, axis=1)

        random_preds = np.random.choice(n_class, len(test_non_featured_idx))

        preds = np.concatenate((featured_preds, random_preds))
        lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls))

        acc = np.sum(preds == lbls) * 1.0 / len(lbls)
        acc_list.append(acc)
    return np.max(acc_list)


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, input, sp_adj, is_sp_fts=False):
        if is_sp_fts:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(
        y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(
        y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva_cluster(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1
