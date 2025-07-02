import math

from torch import optim

from utils.registry import TRAIN_REGISTRY
from utils.misc import *
from utils.loss_funcs import *
from utils.evaluation import *
from sklearn.cluster import KMeans
import os
import numpy as np


@TRAIN_REGISTRY.register()
def Train_EM(args, model, pyg_graph, train_fts_idx, vali_test_idx):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    input_features = pyg_graph.x
    input_features[vali_test_idx] = 0.0
    input_features = sp.csr_matrix(input_features.numpy()).toarray()

    pyg_graph.x = torch.FloatTensor(input_features)

    mask = MaskEdge(p=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_graph = pyg_graph.to(device)

    ##################

    print(args)
    print("attribute:", pyg_graph.x.shape)
    print("edge:", pyg_graph.edge_index.shape)
    label = pyg_graph.y

    print("Initial Evaluation...")
    cluster_labels, clusters, acc, nmi, ari, f1 = clustering_evaluation(args, pyg_graph, model, label, args.c_init,
                                                                        flag='clustering_evaluation')
    print('==>[clustering results]')
    print(f'acc={acc:.4f}, nmi={nmi:.4f}, ari={ari:.4f}, f1={f1:.4f}')

    if not os.path.exists(os.path.join(os.getcwd(), 'save', 'model_param', args.method)):
        os.makedirs(os.path.join(os.getcwd(), 'save',
                                 'model_param', args.method))

    print("start training!")

    for epoch in range(args.epoch):

        if (epoch + 1) % args.cluster_updating == 0:
            cluster_labels, clusters = clustering_evaluation(args, pyg_graph, model, label, args.c_init,
                                                             flag='clustering', epoch=epoch + 1)

        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        Z, edge_pos_output, edge_neg_output = model(pyg_graph, mask, cluster_labels, clusters, train_fts_idx,
                                                    vali_test_idx)

        torch.autograd.set_detect_anomaly(True)
        clusters1 = compute_centers(args, Z, cluster_labels)
        L_c = compute_cluster_loss_no_contrastive(clusters, clusters1, cluster_labels, args.temperature, args.num_class)
        L_e = ce_loss(edge_pos_output, edge_neg_output)
        loss_train = L_c + L_e

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        acc, nmi, ari, f1 = clustering_evaluation(args, pyg_graph, model, label, args.c_init, flag='evaluation',
                                                  epoch=epoch + 1)
        if epoch == (args.epoch - 1):
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                        'best_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))

    kmeans = KMeans(n_clusters=args.num_class, n_init=args.c_init)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                  'best_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    model.eval()
    with torch.no_grad():
        feat = model.embed(pyg_graph)
        feat = nn.functional.normalize(feat, dim=1)
    cluster_labels = kmeans.fit_predict(feat.data.cpu().numpy())

    acc, nmi, ari, f1 = eva_cluster(label.cpu().numpy(), cluster_labels, 0)

    return acc, nmi, ari, f1


def clustering_evaluation(args, pyg_graph, model, label, iterations, flag='1', epoch=0):
    kmeans = KMeans(n_clusters=args.num_class, n_init=iterations)
    model.eval()
    with torch.no_grad():
        feat = model.embed(pyg_graph)
        feat = nn.functional.normalize(feat, dim=1)

    cluster_labels = kmeans.fit_predict(feat.data.cpu().numpy())

    clusters = compute_centers(args, feat, cluster_labels)

    if flag == 'evaluation':
        acc, nmi, ari, f1 = eva_cluster(label.cpu().numpy(), cluster_labels, epoch)
        return acc, nmi, ari, f1
    elif flag == 'clustering':
        print('updata')
        return cluster_labels, clusters
    else:
        acc, nmi, ari, f1 = eva_cluster(label.cpu().numpy(), cluster_labels, epoch)

        return cluster_labels, clusters, acc, nmi, ari, f1


def compute_centers(args, x, cluster_labels):
    n_samples = x.size(0)
    if len(torch.from_numpy(cluster_labels).size()) > 1:
        weight = cluster_labels.T
    else:
        weight = torch.zeros(args.num_class, n_samples).to(x)
        weight[cluster_labels, torch.arange(n_samples)] = 1
        a = weight
    weight = F.normalize(weight, p=1, dim=1)
    centers = torch.mm(weight, x)
    centers = F.normalize(centers, dim=1)
    return centers


def compute_cluster_loss_no_contrastive(q_centers,
                                        k_centers,
                                        psedo_labels,
                                        temperature,
                                        num_cluster):
    pos = F.cosine_similarity(q_centers, k_centers).mean()

    n = q_centers.shape[0]
    neg = 0.0

    for i in range(n):
        arr = np.arange(n)
        arr = np.delete(arr, i)
        random_element = np.random.choice(arr)
        neg_i = F.cosine_similarity(q_centers[i].unsqueeze(dim=0), k_centers[random_element].unsqueeze(dim=0))

        neg += neg_i
    loss = neg / n - pos
    return loss


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
