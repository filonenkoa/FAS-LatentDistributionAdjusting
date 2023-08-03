from box import Box
import torch


def l2_norm(input, dim):
    norm = torch.norm(input, p=2, dim=dim, keepdim=True)
    return torch.div(input, norm)


def cos_simularity(embedding, prototype, tau=10):
    norm_embedding = l2_norm(embedding, 1)
    norm_prototype = l2_norm(prototype, 0)
    cos_dist = torch.mm(norm_embedding, norm_prototype)
    exp_cos_dist = torch.exp(cos_dist/tau)
    cos_den = torch.sum(exp_cos_dist, dim=1, keepdim=True)
    exp_cos_dist = torch.div(exp_cos_dist, cos_den)
    cos_dist = torch.mul(cos_dist, exp_cos_dist)
    return torch.sum(cos_dist, dim=1)


def get_backbone(config: Box) -> torch.nn.Module:
    if config.model.base == "efficientformerv2_s0":
        from efficientformer import EFFICIENTFORMER_V2_S0 as Backbone
    elif config.model.base == "efficientformerv2_s1":
        from efficientformer import EFFICIENTFORMER_V2_S1 as Backbone
    else:
        raise NotImplementedError
    return Backbone(config)