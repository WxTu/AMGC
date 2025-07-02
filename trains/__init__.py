from utils.registry import TRAIN_REGISTRY
from utils.misc import scandir
import importlib
from copy import deepcopy
from os import path as osp

__all__ = ['pre_train']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'trains.{file_name}') for file_name in model_filenames]

def pre_train(args):
    """Build model from options.
    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    if args.method in ['GCN', 'GAT', 'GraphSAGE']:
        train_func = TRAIN_REGISTRY.get("Train_base_gnn")
    else:
        opt = deepcopy(vars(args))
        train_func = TRAIN_REGISTRY.get("Train_"+ opt['method'])
    print(f'Function [{train_func.__name__}] is created.')
    return train_func
