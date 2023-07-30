from box import Box
import torch
import torch.nn as nn
from models import get_backbone
from models.utils import cos_simularity
from models.loss import *
from reporting import report


class LDAModel(nn.Module):        
    def __init__(self, config: Box) -> None:
        super().__init__(config=config)
        self.backbone: nn.Module = get_backbone(config)
        self.classifier = nn.Linear(config.model.descriptor_size, config.model.num_classes)
        self.pos_prototype = nn.Parameter(torch.rand(config.model.descriptor_size, config.model.num_prototypes))
        self.neg_prototype = nn.Parameter(torch.rand(config.model.descriptor_size, config.model.num_prototypes))
        self.self_check()

    def forward(self, x):
        x = self.backbone(x)
        pos_dist = torch.acos(cos_simularity(x, self.pos_prototype))
        neg_dist = torch.acos(cos_simularity(x, self.neg_prototype))
        x = self.classifier(x)
        return x, torch.stack((neg_dist, pos_dist), dim=0)

    def read_prototype(self):
        return self.pos_prototype, self.neg_prototype
    
    def self_check(self):
        x = torch.rand(1, 3, 224, 224)
        y = torch.tensor([1])
        cls_loss = nn.CrossEntropyLoss()
        inter_loss = InterLoss(delta=0.5)
        intra_loss = IntraLoss(delta=0.5)
        data_loss = DataLoss(scale=2, margin=0.5)
        y_hat, dist = self.forward(x)
        pos, neg = self.read_prototype()
        _ = cls_loss(y_hat, y)
        _ = inter_loss(pos, neg)
        _ = intra_loss(pos, neg)
        _ = data_loss(dist, y)
        report(f'Rank {self.config.world_rank}: Model feedforward check passed.')

