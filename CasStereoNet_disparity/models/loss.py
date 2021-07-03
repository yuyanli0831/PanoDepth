import torch.nn.functional as F
import torch

def model_gwcnet_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def model_psmnet_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def calculate_berhu_loss(pred, gt, mask, weights):
    diff = gt - pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (diff**2 + c**2) / (2 * c)
    loss = leq * abs_diff + (1 - leq) * l2_losses
    #_, c, __, ___ = loss.size()
    count = torch.sum(mask, dim=[1, 2], keepdim=True).float()
    masked_loss = loss * mask.float()
    weighted_loss = masked_loss * weights

    return torch.mean(torch.sum(weighted_loss, dim=[1, 2], keepdim=True) / count)

def stereo_psmnet_loss(inputs, target, mask, **kwargs):

    disp_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=target.dtype, device=target.device, requires_grad=False)
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        disp0, disp1, disp2, disp3 = stage_inputs["pred0"], stage_inputs["pred1"], stage_inputs["pred2"], stage_inputs["pred3"]
        #disp0, disp1, disp2 = stage_inputs["pred0"], stage_inputs["pred1"], stage_inputs["pred2"]
        
        loss = 0.5 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
               0.5 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
               0.7 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean') + \
               1.0 * F.smooth_l1_loss(disp3[mask], target[mask], reduction='mean')
        '''
        weights = torch.ones_like(target, dtype=torch.float32)
        
        loss = 0.5 * calculate_berhu_loss(disp0, target, mask, weights) + \
               0.5 * calculate_berhu_loss(disp1, target, mask, weights) + \
               0.7 * calculate_berhu_loss(disp2, target, mask, weights) + \
               1.0 * calculate_berhu_loss(disp3, target, mask, weights)
        '''
        if disp_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += disp_loss_weights[stage_idx] * loss
        else:
            total_loss += loss

    return total_loss