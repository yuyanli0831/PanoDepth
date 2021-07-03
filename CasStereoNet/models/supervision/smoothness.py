import torch

def guided_smoothness_loss(input_duv, guide_duv, mask):
    guidance_weights = torch.exp(-guide_duv)
    smoothness = input_duv * guidance_weights
    smoothness[~mask] = 0.0
    return torch.sum(smoothness) / torch.sum(mask)