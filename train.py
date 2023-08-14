from argparse import ArgumentParser
import datetime
import math

import os
import os.path as osp
from pathlib import Path
import shutil
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
from box import Box
from loguru import logger
from albumentations import Compose
from dataset import ConcatDatasetWithLabels, FASDataset
from models import build_network, load_checkpoint
from optimizers import get_optimizer

from reporting import report
from samplers import DistributedBalancedSampler
from schedulers import init_scheduler
from trainer import Trainer
from transforms import get_transforms


def get_config() -> Box:
    parser = ArgumentParser(description='Latent Distribution Adjusting for Face Anti-Spoofing training')
    parser.add_argument('--config', type=str, help='Name of the configuration (.yaml) file')
    args = parser.parse_args()
    config_path = Path("configs", args.config)
    assert config_path.is_file(), f"Configuration file {config_path} does not exist"
    config = read_cfg(cfg_file=config_path.as_posix())
    config.config_path = config_path
    return config


def read_cfg(cfg_file: str) -> Box:
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (Box): configuration in Box dict wrapper
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return Box(cfg)


def init_libraries(config: Box) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Notice: results won't be reproducible on different hardware
    torch.backends.cudnn.benchmark = True

    if config.fp16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    config.device = torch.device("cuda")
    config.device_name = "cuda"  # hardcoded, no CPU/HPU support


def init_communication(config: Box) -> None:
    world_size = os.environ.get("WORLD_SIZE")
    if not world_size:
        config.world_size = 1
        config.local_rank = 0
        config.world_rank = 0
        config.dist_url = "127.0.0.1:23001"
    else:
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        config.world_rank = int(os.environ["RANK"])
        config.dist_url = f'{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    
        dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(seconds=config.dist_timeout),
                init_method=f"tcp://{config.dist_url}",
                world_size=config.world_size,
                rank=config.world_rank
                )
    torch.cuda.set_device(f"cuda:{config.local_rank}")
    # report(f"Rank {config.local_rank}. Set device {config.device}")
    
    report(f"DEVICE: {config.device_name}\tWORLD: {config.world_size}\tLOCAL_RANK: {config.local_rank}\tWORLD_RANK: {config.world_rank}")
    os.environ["ID"] = str(config.local_rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)



def init_logger(config: Box) -> None:
    logger.add(Path(config.log_dir, "log.log"))


def get_dataloaders(config: Box) -> Tuple[DataLoader, List[DataLoader]]:
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    train_dataset = get_train_set(config, train_transform)
    val_datasets = get_val_sets(config, val_transform)

    sampler = None
    
    if config.train.balanced_sampler:
        sampler = DistributedBalancedSampler(
                dataset=train_dataset,
                num_replicas=config.world_size,
                rank=config.world_rank,
                shuffle=True,
                seed=config.seed,
                drop_last=False)
    elif config.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset = train_dataset,
            shuffle = True,
            seed=config.seed,
            drop_last=True
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True if sampler is None else False,
        num_workers=config.dataset.num_workers,
        sampler=sampler
    )
    val_loaders = []
    for ds in val_datasets:
        val_loaders.append(
            DataLoader(
                dataset=ds,
                batch_size=config['val']['batch_size'],
                shuffle=False,
                num_workers=config.dataset.num_workers_val,
                drop_last=False
                ))
    
    return train_loader, val_loaders


def confirm_dataset_path(dataset_path: Path) -> Path:
    if not dataset_path.is_file():
        possible_markup_path = dataset_path / "markup.csv"
        if possible_markup_path.is_file():
            dataset_path = possible_markup_path
        else:
            raise Exception(f"Could not find markup file for {dataset_path}")
    return dataset_path


def initialize_datasets(dataset_paths: List[str], transforms: Compose, smoothing: bool, is_train: bool) -> List[FASDataset]:
    datasets = []
    for dataset_path in dataset_paths:
        dataset_path = confirm_dataset_path(Path(dataset_path))
        datasets.append(FASDataset(
            root_dir=dataset_path.parent,
            csv_path=dataset_path,
            transform=transforms,
            smoothing=smoothing,
            is_train=is_train
        ))
    return datasets
            

def get_train_set(config: Box, transforms: Compose) -> ConcatDataset:
    assert len(config.dataset.train_set) > 0
    if config.world_rank == 0:
        logger.info(f"Combining {config.dataset.train_set} train datasets")
    datasets = ConcatDatasetWithLabels(
        initialize_datasets(
            config.dataset.train_set, transforms, config.dataset.smoothing, True
            )
        )
    return datasets
    

def get_val_sets(config: Box, transforms: Compose) -> List[FASDataset]:
    assert len(config.dataset.val_set) > 0
    max_datasets_per_rank = math.ceil(len(config.dataset.val_set) / config.world_size)
    lower_bound = config.world_rank * max_datasets_per_rank
    upper_bound = min((config.world_rank + 1) * max_datasets_per_rank, len(config.dataset.val_set))
    local_datasets = config.dataset.val_set[lower_bound:upper_bound]
    config.datasets_start_index = lower_bound
    config.all_dataset_names = [FASDataset.path_to_name(p) for p in config.dataset.val_set]
    report(f"Rank {config.world_rank}: Local validation datasets: {[FASDataset.path_to_name(ds) for ds in local_datasets]}")
    datasets = initialize_datasets(local_datasets, transforms, config.dataset.smoothing, False)
    return datasets



def cli_main():
    config: Box = get_config()
    init_libraries(config)
    init_communication(config)
    
    if config.world_rank == 0:
        current_datetime = datetime.datetime.now()
        config.log_dir = Path(config.log_root, f"{config.model.base}_{config.dataset.name}", str(current_datetime).replace(":", "-"))
        config.log_dir.mkdir(parents=True)
        init_logger(config)
        shutil.copy(config.config_path, Path(config.log_dir, "config.yaml"))

    if config.local_rank == 0:
        logger.info(config)
    
    train_loader, val_loaders = get_dataloaders(config)

    # build model and engine
    state_dict = load_checkpoint(config)
    model = build_network(config, state_dict)
    model.to(config.device)
    optimizer = get_optimizer(config, model, state_dict.get("optimizer"))
    lr_scheduler, is_batch_scheduler = init_scheduler(config, optimizer, state_dict.get("scheduler"))
    
    writer = SummaryWriter(config.log_dir) if config.world_rank == 0 else None
    
    start_epoch = state_dict.get("epoch") + 1 if "epoch" in state_dict.keys() else 0
    
    trainer = Trainer(
        config=config,
        network=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        is_batch_scheduler=is_batch_scheduler,
        device=config.device,
        trainloader=train_loader,
        val_loaders=val_loaders,
        writer=writer,
        start_epoch=start_epoch
    )

    logger.info(f"Rank {config.world_rank}. Start training...")
    trainer.train()
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    cli_main()

