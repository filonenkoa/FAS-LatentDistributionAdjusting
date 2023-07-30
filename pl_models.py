import torch
import torch.nn as nn
from models.LDA import LDAModel
from models.loss import *


class LDA_PL(pl.LightningModule):
    def __init__(self, epoch, batch_size, hyp):
        super().__init__()
        self.save_hyperparameters()
        self.hyp = hyp
        self.n_prototypes = self.hyp['n_prototypes']
        self.batch_size = batch_size
        self.backbone = LDAModel(n_prototypes=self.hyp['n_prototypes'], n_features=self.hyp['n_features'],
                                 n_classes=self.hyp['n_classes'])
        self.epoch = epoch
        self.cls_loss = nn.CrossEntropyLoss()
        self.inter_loss = InterLoss(delta=self.hyp['inter_delta'])
        self.intra_loss = IntraLoss(delta=self.hyp['intra_delta'])
        self.data_loss = DataLoss(scale=self.hyp['scale'], margin=self.hyp['margin'])

    def forward(self, x):
        # use forward for inference/predictions
        y, dist = self.backbone(x)
        return y, dist

    def cal_loss(self, y_hat, dist, y):
        loss = self.cls_loss(y_hat, y)
        pos, neg = self.backbone.read_prototype()
        loss += self.inter_loss(pos, neg) * self.hyp['inter_weight']
        loss += self.intra_loss(pos, neg) * self.hyp['intra_weight'] / (self.n_prototypes * (self.n_prototypes + 1) / 2)
        loss += self.data_loss(dist, y) * self.hyp['data_weight'] / self.batch_size
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, dist = self.backbone(x)
        loss = self.cal_loss(y_hat, dist, y)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.backbone(x)
        loss = self.cls_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.backbone(x)
        loss = self.cls_loss(y_hat, y)
        self.log('test_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hyp['lr'],
                              momentum=self.hyp['momentum'], weight_decay=self.hyp['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=self.hyp['lr_decay'])
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}
        return [opt], [lr_scheduler]
