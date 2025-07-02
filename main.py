from utils.misc import *
from datasets.pyg_data_utils import load_pyg_dataset
from models import build_model
from trains import pre_train
from utils.evaluation import *
import argparse


def build_args():
    parser = argparse.ArgumentParser(
        description="Attrabution-Missing Graph Learning")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--method", type=str, default="EM")
    parser.add_argument("--frame", type=str, default="pyg")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--tsne_seed", type=int, default=0)
    parser.add_argument("--c_init", type=int, default=20)
    parser.add_argument("--split_seed", type=float, default=72)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_fts_ratio", type=float, default=0.4)
    parser.add_argument("--encoder_name", type=str, help='SAT config  GCN/GAT/GraphSAGE', default="GCN")
    parser.add_argument("--dataset_type", type=str, default="one-hot")
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_graph, num_nodes, num_features, num_classes = load_pyg_dataset(args.dataset)

    if args.dataset in ['pubmed']:
        args.dataset_type = 'continuous'
    args.num_nodes = num_nodes
    args.in_features = num_features
    args.num_class = num_classes
    train_id, vali_id, test_id, vali_test_id = data_split(args, num_nodes)

    model = build_model(args)
    model = model.to(device)
    print(args)

    train_func = pre_train(args)
    print('------------------------------start training-------------------------------')
    acc, nmi, ari, f1 = train_func(args, model, pyg_graph, train_id, vali_test_id)

    return acc, nmi, ari, f1


if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, f"configs/{args.method}/configs.yml")
    acc, nmi, ari, f1 = main(args)
