import copy
import time
from box import Box
import torch
from tqdm import tqdm

from reporting import report


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
    elif config.model.base == "fastvit_t8":
        from models.fastvit import FASTVIT_T8 as Backbone
    elif config.model.base == "fastvit_t12":
        from models.fastvit import FASTVIT_T12 as Backbone
    elif config.model.base == "fastvit_s12":
        from models.fastvit import FASTVIT_S12 as Backbone
    else:
        raise NotImplementedError
    return Backbone(config)


def test_inference_speed(input_model, device: str | torch.device = "cpu", input_size: int = 224, iterations: int = 1000):
    # cuDnn configurations
    actual_cuddn_benchmark = torch.backends.cudnn.benchmark
    actual_cudnn_deterministic = torch.backends.cudnn.deterministic
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = copy.deepcopy(input_model).to(device)

    model.eval()

    time_list = []
    for i in tqdm(range(iterations+1), desc="Testing inference time"):
        random_input = torch.randn(1,3,input_size,input_size).to(device)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        model(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        #print(time.time()-tic)
        time_list.append(time.perf_counter()-tic)
    time_list = time_list[1:]
    
    torch.backends.cudnn.benchmark = actual_cuddn_benchmark
    torch.backends.cudnn.deterministic = actual_cudnn_deterministic
    
    return sum(time_list)/iterations
