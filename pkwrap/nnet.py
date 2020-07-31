# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

"""
    Here we implement different features implement in Kaldi nnet3.
    Only those functionalities that cannot be wrapped will be implemented.
"""    

import torch

def max_change(model, max_param_change=2.0, max_change_scale=1.0, scale=1.0):
    """Untested implementation of max_param_change in Kaldi"""
    scale_factors = []
    num_components_updated = 0
    for i, p in enumerate(model.parameters()):
        if i == 0:
            device = p.device
            max_param_change = torch.tensor(max_param_change, device=device, requires_grad=False)
            max_change_scale = torch.tensor(max_change_scale, device=device, requires_grad=False)
            parameter_delta_sq = torch.tensor(0.0, device=device, requires_grad=False)
            scale = torch.tensor(scale, device=device, requires_grad=False)
        dp2 = (p.grad.data*p.grad.data).sum()
        dp = dp2.pow(0.5)
        if len(p.shape) == 2 and p.shape[1] == 2344:
            max_change = torch.tensor(1.5, device=device, requires_grad=False)
        elif len(p.shape) == 1 and p.shape[0] == 2344:
            max_change = torch.tensor(1.5, device=device, requires_grad=False)
        else:
            max_change = torch.tensor(0.75, device=device, requires_grad=False)
        if dp*scale> max_change*max_change_scale:
            sf = (max_change*max_change_scale)/(dp*scale)
            scale_factors.append(sf)
            num_components_updated += 1
        else:
            scale_factors.append(torch.tensor(1.0, device=device))
        parameter_delta_sq += scale_factors[-1].pow(2.0)*dp2
    parameter_delta = torch.sqrt(parameter_delta_sq)*torch.abs(scale)
    print("Parameter delta", parameter_delta)
    assert not torch.isnan(parameter_delta)
    assert not torch.isinf(parameter_delta)
    if parameter_delta>max_param_change*max_change_scale:
        scale.mul_(max_param_change*max_change_scale/parameter_delta)
    print("Scale ", scale, num_components_updated)
    for sf in scale_factors:
        sf.mul_(scale)
    print("Factors ", scale_factors)
    for sf, p in zip(scale_factors, model.parameters()):
        p.grad.data.mul_(sf)

