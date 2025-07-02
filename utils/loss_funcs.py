import torch
import torch.nn.functional as F

def loss_function_discrete(output_fts, fts_labls, pos_weight_tensor, neg_weight_tensor):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(output_fts, shape=[-1])
    fts_labls_reshape = torch.reshape(fts_labls, shape=[-1])
    weight_mask = torch.where(fts_labls_reshape != 0.0, pos_weight_tensor, neg_weight_tensor)
    loss_bce = torch.mean(BCE(output_fts_reshape, fts_labls_reshape) * weight_mask)
    return loss_bce

def loss_function_continuous(output_fts, fts_labls, pos_weight_tensor, neg_weight_tensor):
    MSE = torch.nn.MSELoss(reduction='none')
    output_fts_reshape = torch.reshape(output_fts, shape=[-1])
    fts_labls_reshape = torch.reshape(fts_labls, shape=[-1])
    loss_mse = torch.mean(MSE(output_fts_reshape, fts_labls_reshape))
    return loss_mse

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss