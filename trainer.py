import os
from pathlib import Path
from random import randint
import sys
from typing import Dict, List
from box import Box
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from containers import ClassificationMetrics, PredictionCounters
from base_trainer import BaseTrainer
from meter import AvgMeter
from tqdm.auto import tqdm, trange
import time
import datetime

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from metrics_counter import MetricsCounter
from reporting import report
from models.LDA import InterLoss, IntraLoss, DataLoss
from containers import BestMetric


class Trainer(BaseTrainer):
    def __init__(self,
                 config: Box,
                 network: Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 is_batch_scheduler: bool,
                 device: torch.device,
                 trainloader: DataLoader,
                 val_loaders: List[DataLoader],
                 writer: SummaryWriter,
                 start_epoch: int = 0):
        super().__init__(config, network, optimizer, lr_scheduler, is_batch_scheduler, device, trainloader, val_loaders, writer)
        # self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.train_loader), per_iter_vis=True)
        self.start_epoch = start_epoch
        
        self.epoch_time = AvgMeter(writer=writer, name="Epoch time, s", num_iter_per_epoch=1)
        if self.config.local_rank == 0:
            self.best_metrics = Box({
                "acer": BestMetric(1.0, -1),
                "f3": BestMetric(0.0, -1),
                "f1": BestMetric(0.0, -1)
            })
        
        self.metrics_counter = MetricsCounter()
        
        self.total_val_sets = len(self.config.dataset.val_set)
        
        self.cls_loss = CrossEntropyLoss()
        self.inter_loss = InterLoss(delta=config.loss.inter_delta)
        self.intra_loss = IntraLoss(delta=config.loss.intra_delta)
        self.data_loss = DataLoss(scale=config.loss.scale, margin=config.loss.margin)
        
        
    def save_model(self, epoch, val_metrics: ClassificationMetrics):
        file_name = Path(self.config.log_dir, f"{epoch:04d}_{self.config.model.base}_{val_metrics.acer:.4f}.pth")
        
        model = self.network.module if dist.is_initialized() and self.config.world_size > 1 else self.network

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
      }
        
        torch.save(state, file_name.as_posix())

    @staticmethod
    def update_prediction_counters(prediction_counters: PredictionCounters,
                                   prediction: torch.Tensor,
                                   target: torch.Tensor) -> None:
        predicted_class = torch.argmax(prediction, dim=1)
        # TODO: Think how to vectorize the computations
        for pred, t in zip(predicted_class, target):
            if pred == t:
                if pred == 0:
                    prediction_counters.tp += 1
                else:
                    prediction_counters.tn += 1
            else:
                if pred == 0:
                    prediction_counters.fp += 1
                else:
                    prediction_counters.fn += 1
        
    def train_one_epoch(self, epoch):
        prediction_counters_epoch = PredictionCounters()
        self.network.train()
        self.train_loss_metric.reset(epoch)
        max_num_batches = len(self.train_loader) // self.config.world_size
        
        iterator = enumerate(self.train_loader)
        if self.config.local_rank == 0:
            iterator = tqdm(iterator, desc=f"Training epoch {epoch}", total=max_num_batches)
            
        for batch_index, (img, label) in iterator:
            if self.lr_scheduler is not None and self.is_batch_scheduler:
                self.lr_scheduler.step(epoch + (batch_index / max_num_batches))
            img, label = img.to(self.device), label.to(self.device)
            model = self.network.module if dist.is_initialized() and self.config.world_size > 1 else self.network
            
            prediction, distance = model(img)
            
            loss = self.cls_loss(prediction, label)
            pos, neg = model.read_prototype()
            loss += self.inter_loss(pos, neg) * self.config.loss.inter_weight
            loss += self.intra_loss(pos, neg) * self.config.loss.intra_weight / (self.config.model.num_prototypes * (self.config.model.num_prototypes + 1) / 2)
            loss += self.data_loss(distance, label) * self.config.loss.data_weight / img.shape[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            prediction_counters_batch = PredictionCounters()
            self.update_prediction_counters(prediction_counters_batch, prediction, label)
            prediction_counters_epoch += prediction_counters_batch
            
            # Update metrics
            # batch_metrics = self.metrics_counter(prediction_counters_batch)
            epoch_metrics = self.metrics_counter(prediction_counters_epoch)
            
            self.train_loss_metric.update(loss.item())
            if self.config.world_rank == 0:
                lr = self.get_lr()
                text = (
                    f"E: {epoch}, "
                    f"loss: {self.train_loss_metric.avg:.4f}, "
                    f"ACER: {epoch_metrics.acer*100:.2f}%, "
                    f"F1: {epoch_metrics.f1*100:.2f}%, "
                    f"F3: {epoch_metrics.f3*100:.2f}%, "
                    f"P: {epoch_metrics.precision*100:.2f}%, "
                    f"R: {epoch_metrics.recall*100:.2f}%, "
                    f"S: {epoch_metrics.specificity*100:.2f}%, "
                    f"LR: {lr:.4E}"
                    )
                iterator.set_description(text)
                globiter = epoch * max_num_batches + batch_index
                self.writer.add_scalar("LR", lr, globiter)
                self.writer.add_scalar("Loss", loss.item(), globiter) 
                   
        if self.config.world_rank == 0:
            epoch_report = f"\nEpoch {epoch}, train metrics:\n{epoch_metrics}"
            report(epoch_report, use_telegram=self.config.telegram_reports)
            self.writer.add_scalar("ACER/train", epoch_metrics.acer, epoch)
            self.writer.add_scalar("APCER/train", epoch_metrics.apcer, epoch)
            self.writer.add_scalar("BPCER/train", epoch_metrics.bpcer, epoch)
            self.writer.add_scalar("F1/train", epoch_metrics.f1, epoch)
            self.writer.add_scalar("F3/train", epoch_metrics.f3, epoch)
            self.writer.add_scalar("Precision/train", epoch_metrics.precision, epoch)
            self.writer.add_scalar("Recall/train", epoch_metrics.recall, epoch)
            self.writer.add_scalar("Specificity/train", epoch_metrics.specificity, epoch)
            
        if self.lr_scheduler is not None and not self.is_batch_scheduler:
            self.lr_scheduler.step()
        torch.cuda.empty_cache()
        
    def get_lr(self) -> float:
        current_lr = 0
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        return current_lr


    def update_best_metrics(self, val_metrics: Dict[str, ClassificationMetrics], epoch: int) -> None:
        if self.config.local_rank == 0:
            print("!!!!!!!!!!!!", self.best_metrics)
            if float(val_metrics["total"].acer) < self.best_metrics.acer():
                report(f"Validation ACER improved from {self.best_metrics.acer()*100:.4f}% to {val_metrics['total'].acer*100:.4f}%")
                self.best_metrics.acer = BestMetric(val_metrics["total"].acer, epoch)
            if float(val_metrics["total"].f3) > self.best_metrics.f3():
                report(f"Validation F3 improved from {self.best_metrics.f3()*100:.4f}% to {val_metrics['total'].f3*100:.4f}%")
                self.best_metrics.f3 = BestMetric(val_metrics["total"].f3, epoch)
            if float(val_metrics["total"].f1) > self.best_metrics.f1():
                report(f"Validation F1 improved from {self.best_metrics.f1()*100:.4f}% to {val_metrics['total'].f1*100:.4f}%")
                self.best_metrics.f1 = BestMetric(val_metrics["total"].f1, epoch)
            

            
    def train(self):
        if self.config.train.val_before_train:
            epoch = -1
            val_metrics = self.validate(epoch)
            self.update_best_metrics(val_metrics, epoch)
        if dist.is_initialized():
            dist.barrier()
        
        iterator = range(self.start_epoch, self.config.train.num_epochs, 1)
        if self.config.local_rank == 0:
            iterator = tqdm(iterator)
        
        for epoch in iterator:
            epoch_start_time = time.time()
            
            if self.config.local_rank == 0:
                iterator.set_description(f"Epoch {epoch}")
            
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if dist.is_initialized():
                dist.barrier()
            val_metrics = self.validate(epoch)
            if self.config.world_rank == 0:
                self.update_best_metrics(val_metrics, epoch)
                
                self.save_model(epoch, val_metrics["total"])
                epoch_time = time.time() - epoch_start_time
                self.epoch_time.update(epoch_time)
                epoch_end_message = (
                    f"Epoch {epoch} time = {int(self.epoch_time.val)} seconds",
                    f"Best F3 = {self.best_metrics.f3()*100:.4f}% at epoch {self.best_metrics.f3.epoch}",
                    f"Best F1 = {self.best_metrics.f1()*100:.4f}% at epoch {self.best_metrics.f1.epoch}",
                    f"Best ACER = {self.best_metrics.acer()*100:.4f}% at epoch {self.best_metrics.acer.epoch}"
                    )
                report(epoch_end_message, use_telegram=self.config.telegram_reports)
    
    @staticmethod                 
    def estimate_time(start_time: float, cur_iter: int, max_iter: int):
        telapsed = time.time() - start_time
        testimated = (telapsed/cur_iter)*(max_iter)

        finishtime = start_time + testimated
        finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

        lefttime = testimated-telapsed  # in seconds

        return (int(telapsed), int(lefttime), finishtime)
            
    def validate(self, epoch) -> Dict[str, ClassificationMetrics]:
        self.network.eval()
        prediction_counters = {val_loader.dataset.name: PredictionCounters() for val_loader in self.val_loaders}
        tqdm_desc = f"Rank {self.config.world_rank}: Validating epoch {epoch}"
        tqdm_position = self.config.world_rank + 1
        tqdm_pbar = tqdm(enumerate(self.val_loaders), desc=tqdm_desc, position=tqdm_position, total=len(self.val_loaders))
        
        if self.config.world_size > 1:  # collect prediction counters
            predictions_tensor = torch.zeros(self.total_val_sets * 4, dtype=torch.int64, device=self.config.device)
            
        num_batches = 0
        for val_loader in self.val_loaders:
            num_batches += len(val_loader)
        start_time = time.time()
        batch_index_acc = 0  # cumulative index for all val sets
        with torch.no_grad():
            for i, val_loader in tqdm_pbar:
                for img, label in val_loader:
                    img, label = img.to(self.device), label.to(self.device)
                    model = self.network.module if dist.is_initialized() and self.config.world_size > 1 else self.network
                    prediction, distance = model(img)

                    prediction_counters_batch = PredictionCounters()
                    label_squeezed = label.squeeze().type(torch.int8)
                    self.update_prediction_counters(prediction_counters_batch, prediction, label_squeezed)
                    prediction_counters[val_loader.dataset.name] += prediction_counters_batch
                    batch_index_acc += 1
                    time_elapsed, time_left, time_eta = self.estimate_time(start_time, batch_index_acc, num_batches)
                    tqdm_pbar.set_description(tqdm_desc + f" {val_loader.dataset.name} {batch_index_acc}/{num_batches} ETA:{time_left}")
                    
                if self.config.world_size > 1:
                    position = (self.config.datasets_start_index + i) * 4
                    predictions_tensor[position:position + 4] = prediction_counters[val_loader.dataset.name].as_tensor()

        # Gather predictions
        if self.config.world_size > 1:
            dist.reduce(predictions_tensor, dst=0, op=dist.ReduceOp.SUM)
            for dataset_index, dataset_name in enumerate(self.config.all_dataset_names):
                lower_bound = dataset_index * 4
                dataset_tensor = predictions_tensor[lower_bound:lower_bound + 4]
                prediction_counters[dataset_name] = PredictionCounters.from_tensor(dataset_tensor)
        metrics = dict()
        if self.config.world_rank == 0:
            counters_sum = PredictionCounters()
            val_end_text = f"\nValidation epoch {epoch}"
            for dataset_name, counters in prediction_counters.items():
                metrics[dataset_name] = self.metrics_counter(counters)
                counters_sum += counters
                val_end_text += f"\nDataset {dataset_name}: {metrics[dataset_name]}"
                self.writer.add_scalar(f"ACER/{dataset_name}", metrics[dataset_name].acer, epoch)
                self.writer.add_scalar(f"APCER/{dataset_name}", metrics[dataset_name].apcer, epoch)
                self.writer.add_scalar(f"BPCER/{dataset_name}", metrics[dataset_name].bpcer, epoch)
                self.writer.add_scalar(f"F1/{dataset_name}", metrics[dataset_name].f1, epoch)
                self.writer.add_scalar(f"F3/{dataset_name}", metrics[dataset_name].f3, epoch)
                self.writer.add_scalar(f"Precision/{dataset_name}", metrics[dataset_name].precision, epoch)
                self.writer.add_scalar(f"Recall/{dataset_name}", metrics[dataset_name].recall, epoch)
                self.writer.add_scalar(f"Specificity/{dataset_name}", metrics[dataset_name].specificity, epoch)
                
                
            metrics["total"] = self.metrics_counter(counters_sum)
            val_end_text += f"\nCombined metrics: {metrics['total']}"
            
            self.writer.add_scalar(f"ACER/val", metrics["total"].acer, epoch)
            self.writer.add_scalar(f"APCER/val", metrics["total"].apcer, epoch)
            self.writer.add_scalar(f"BPCER/val", metrics["total"].bpcer, epoch)
            self.writer.add_scalar(f"F1/val", metrics["total"].f1, epoch)
            self.writer.add_scalar(f"F3/val", metrics["total"].f3, epoch)
            self.writer.add_scalar(f"Precision/val", metrics["total"].precision, epoch)
            self.writer.add_scalar(f"Recall/val", metrics["total"].recall, epoch)
            self.writer.add_scalar(f"Specificity/val", metrics["total"].specificity, epoch)    
                
            report(val_end_text, use_telegram=self.config.telegram_reports)
        
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        return metrics
                