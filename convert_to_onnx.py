import argparse
from collections import defaultdict
from typing import List
from pathlib import Path
import logging
import sys
import torch
import math
from PIL import Image
import numpy as np
import onnxruntime as ort
import pandas as pd
from reporting import report


sys.path.append(Path(__file__).absolute().parent.parent.as_posix())
from models import build_network, load_checkpoint
from train import read_cfg


def convert_to_onnx(config, onnx_path: Path, batch_size: int = 1):   
    batch_pt = torch.rand(size=(batch_size, 3, config.dataset.crop_size, config.dataset.crop_size))
    batch_onnx = batch_pt.clone().cpu().numpy()
   
    output_raw = model(batch_pt)

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            model_traced = torch.jit.trace(
                model,
                example_inputs=batch_pt,
                check_trace=True,
                check_inputs=[batch_pt],
                strict=True
                )

    if batch_size == 1:
        dynamic_axes = None
    else:
        dynamic_axes = {'input': {0: 'batch_size'},
                      'spoof': {0: 'batch_size'}}

    torch.onnx.export(
        model=model_traced,
        args=batch_pt,
        f=onnx_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['spoof'],
        dynamic_axes=dynamic_axes
        )

    model_onnx = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
    input_name = model_onnx.get_inputs()[0].name
    output_onnx = model_onnx.run(None, {input_name: batch_onnx.astype(np.float32)})
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output_raw), output_onnx[0], rtol=1e-03, atol=1e-05)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LDA FAS, convet to ONNX')
    parser.add_argument('--config', type=str, help='Path to the configuration (.yaml) file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.pth) path')
    args = parser.parse_args()
    report(f"Converting the checkpoint {args.checkpoint} with the config {args.config}")
    config = read_cfg(cfg_file=args.config)
    config.model.checkpoint_path = args.checkpoint
    state_dict = load_checkpoint(config)
    config.world_rank = 0
    config.world_size = 1
    config.model.pretrained = False
    config.device = "cpu"
    model = build_network(config, state_dict)
    model.inference_mode = True
    if model.can_reparameterize:
        model.reparameterize()  # should be called in the train mode
    model.eval()
    
    onnx_path = Path(args.checkpoint).parent / Path(f"{Path(args.checkpoint).stem}_dyn.onnx")
    report("Converting with dynamic batch size")
    convert_to_onnx(config, onnx_path, batch_size=4)
    report("Done")