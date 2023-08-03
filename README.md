
## Latent Distribution Adjusting

This is an extension of the [unofficial implementation](https://github.com/RicardooYoung/LatentDistributionAdjusting) of [**Latent Distribution Adjusting for Face Anti-Spoofing**](https://arxiv.org/abs/2305.09285) using PyTorch.
The code base is rewritten from PyTorch Lightning to vanilla PyTorch to ensure the better control of everything.
  
## Improvements to the original repository
- [x] Transfer from PyTorch Lightning to vanilla PyTorch
- [x] EfficientFormerV2 support
- [x] Use config files
- [x] Data augmentations
- [x] TurboJPEG support for faster image decoding
- [x] Multiple datasets training
- [x] Compute FAS-related metrics (ACER, etc.)
- [x] Telegram reports
- [x] Compute metrics for each val dataset separately
- [x] Split validation into miltiple GPUs
- [x] Balanced sampler suitable for DDP
- [ ] FP16/AMP support
- [ ] Conversion to ONNX

## How to use

### Installation
```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

### Pretrained weights
When using EfficientFormerV2 model, put pretrained weights to `weights/efficientformerv2`


### Telegram reports
This repository supports sending messages after each epoch to a telegram bot.\
To make it work, create `.env` file with Telegram bot token and chat ids:
```bash
$ cp .env.template .env
$ nano .env
```

### Data preparation
```
datasets
    |---images
    |     |--img1
    |     |--img2
    |     |...
    |---train.csv
    |---val.csv
    |---test.csv
```
with [*.csv] having format (label only has 2 classes: 0-Spoofing, 1-Bonafide):
```
image_name      |  label
img_name1.jpg   |    0
img_name2.png   |    1
...
```
`image_name` is the relative path to the image from the locations of the *.csv file.\
One can find utility to convert exisiting images dataset into format supported by current repository in `utils/dataset_preparation/prepare_dataset.py`


### Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config config.yaml
```
`nproc_per_node` is the number of GPUs you want to use.