from torch_geometric.datasets import *
import os

GRAPH_DICT = {
    "cora": Planetoid,
    "citeseer": Planetoid,
    "pubmed": Planetoid,
}

NAME_DICT = {
    'cora': 'Cora',
    'citeseer': "CiteSeer",
    'pubmed': "PubMed",
}


def load_pyg_dataset(dataset_name):
    path = os.path.join(os.getcwd(), 'datasets', 'data', dataset_name)
    Data_Down = GRAPH_DICT[dataset_name]
    Date_Name = NAME_DICT[dataset_name]

    dataset = Data_Down(path, Date_Name)
    data = dataset[0]
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_classes = dataset.num_classes

    return data, num_nodes, num_features, num_classes
