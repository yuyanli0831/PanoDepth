from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import spherical as S360
import supervision as L

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
            #nn.GroupNorm(in_channels//2, in_channels*2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
            #nn.GroupNorm(in_channels//4, in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class feature_extraction(nn.Module):
    def __init__(self, arch_mode="fpn", num_stage=None):
        super(feature_extraction, self).__init__()
        assert arch_mode in ["fpn"]
        self.inplanes = 32
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        #TODO: split modifiy
        self.firstconv_a = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.firstconv_b = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)

        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        #TODO: add inner0 to reduce chs to 32, and memory do not change
        self.inner0 = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=True))

        self.lastconv = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)

        self.out_channels = [32]

        if self.arch_mode == 'unet':
            raise NotImplementedError

        elif self.arch_mode == "fpn":
            final_chs = 32
            if num_stage == 3:
                self.inner1 = nn.Conv2d(32, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(32, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, 16, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, 8, 3, padding=1, bias=False)
                self.out_channels.append(16)
                self.out_channels.append(8)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(32, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, 16, 3, padding=1, bias=False)
                self.out_channels.append(16)


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
                #nn.GroupNorm(planes * block.expansion//4, planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #feature extraction H, W -> H/2, W/2
        output_s1   = self.firstconv_a(x)
        output      = self.firstconv_b(output_s1)
        output_s2   = self.layer1(output)
        #stride 2, H/2, W/2 -> H/4, W/4
        output_raw  = self.layer2(output_s2)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        # pyramid pooling
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=Align_Corners)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=Align_Corners)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=Align_Corners)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=Align_Corners)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        output_msfeat = {}

        output_feature = self.inner0(output_feature)
        out = self.lastconv(output_feature)
        output_msfeat["stage1"] = out
        # stage 1, H/4, W/4
        intra_feat = output_feature

        if self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(output_s2)
                out = self.out2(intra_feat)
                output_msfeat["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(output_s1)
                out = self.out3(intra_feat)
                output_msfeat["stage3"] = out

            elif self.num_stage == 2:
                # stage 2, H/2, W/2
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(output_s2)
                out = self.out2(intra_feat)
                output_msfeat["stage2"] = out

        return output_msfeat

class CostAggregation(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(CostAggregation, self).__init__()

        self.dres0 = nn.Sequential(convbn_3d(in_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1))

        self.dres2 = hourglass(base_channels)

        self.dres3 = hourglass(base_channels)

        self.dres4 = hourglass(base_channels)

        self.classif0 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        self.classif3 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, cost, FineD, FineH, FineW, depth_range_samples, interval, min_depth=0.1, max_depth=8.0):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0) 
        out2 = self.dres3(out1) 
        out3 = self.dres4(out2) 

        cost3 = self.classif3(out3)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.upsample(cost0, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost1 = F.upsample(cost1, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost2 = F.upsample(cost2, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)

            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, interval)
            pred0 = 1/(1/max_depth + (1/min_depth-1/max_depth)*pred0/(FineD-1))

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, interval)
            pred1 = 1/(1/max_depth + (1/min_depth-1/max_depth)*pred1/(FineD-1))

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, interval)
            pred2 = 1/(1/max_depth + (1/min_depth-1/max_depth)*pred2/(FineD-1))

        cost3 = F.upsample(cost3, [FineD, FineH, FineW], mode='trilinear', align_corners=Align_Corners)
        cost3 = torch.squeeze(cost3, 1)
        pred3_prob = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3_prob = disparity_regression(pred3_prob, interval)
        pred3 = 1/(1/max_depth + (1/min_depth-1/max_depth)*pred3_prob/(FineD-1))

        if self.training:
            return torch.abs(pred0), torch.abs(pred1), torch.abs(pred2), torch.abs(pred3)
        else:
            return torch.abs(pred3)

def create_Duv_grid(width, D, data_type=torch.float32):
    height = int(width // 2.0)
    v_range = (
        torch.arange(0, height) # [0 - h]
        .view(1, 1, height, 1) # [1, [0 - h], 1]
        .expand(1, D, height, width) # [1, [0 - h], W]
        .type(data_type)  # [1, H, W]
    )
    u_range = (
        torch.arange(0, width) # [0 - w]
        .view(1, 1, 1, width) # [1, 1, [0 - w]]
        .expand(1, D, height, width) # [1, H, [0 - w]]
        .type(data_type)  # [1, H, W]
    )
    #u_range *= (2 * math.pi / width) # [0, 2 * pi]
    #v_range *= (math.pi / height) # [0, pi]
    D_range = (
        torch.arange(0, D)
        .view(1, D, 1, 1)
        .expand(1, D, height, width) 
        .type(data_type)
    )
    return torch.stack((D_range, u_range, v_range), dim=1)  # [1, 3, D, H, W]

class GetCostVolume(nn.Module):
    def __init__(self):
        super(GetCostVolume, self).__init__()

    def forward(self, x, y, depth_range_samples, ndisp):
        assert (x.is_contiguous() == True)

        bs, channels, height, width = x.size()
        cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()  #(B, 2C, D, H, W)
        sgrid = S360.grid.create_spherical_grid(width).cuda()
        uvgrid = S360.grid.create_image_grid(width, height).cuda()
        
        #up down, depth to disparity    
        disp = torch.stack(
                (
                    torch.zeros_like(depth_range_samples),
                    S360.derivatives.dtheta_vertical(sgrid, depth_range_samples, 0.24)
                ),
                dim=-1
            )  #(B, D, H, W, 2)
               
        uvgrid = uvgrid.permute(0, 2, 3, 1).contiguous().unsqueeze(1).repeat(1, ndisp, 1, 1, 1) 
        # spherical coordinates of the up image 
        up_render_coords = uvgrid + disp
        up_render_coords[torch.isnan(up_render_coords)] = 0.0
        up_render_coords[torch.isinf(up_render_coords)] = 0.0  
        cur_disp_coords_y = up_render_coords[:,:,:,:,1]
        cur_disp_coords_x = up_render_coords[:,:,:,:,0]

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4)   #(B, D, H, W, 2)

        cost[:, x.size()[1]:, :, :, :] = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                                                       padding_mode='zeros').view(bs, channels, ndisp, height, width)

        # a littel difference, no zeros filling
        tmp = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
        # tmp = tmp.transpose(0, 1) #(C, B, D, H, W)
        # #x1 = x2 + d >= d
        # tmp[:, mw < disp_range_samples] = 0
        # tmp = tmp.transpose(0, 1) #(B, C, D, H, W)
        cost[:, :x.size()[1], :, :, :] = tmp
        #duv_grid = create_Duv_grid(width, ndisp).cuda()
        #duv_grid = duv_grid.repeat(bs, 1, 1, 1, 1)
        #cost = torch.cat([cost, duv_grid], 1)
        return cost


class PSMNet(nn.Module):
    def __init__(self, ndisps, disp_interval_pixel, using_ns, ns_size, grad_method="detach",
                 cr_base_chs=[32, 32, 32]):
        super(PSMNet, self).__init__()
        #self.maxdisp = maxdisp
        self.ndisps = ndisps
        self.disp_interval_pixel = disp_interval_pixel
        self.num_stage = len(self.ndisps)
        self.cr_base_chs = cr_base_chs
        self.grad_method = grad_method
        self.ns_size = ns_size
        self.using_ns = using_ns
        #assert self.maxdisp == 192
        assert self.grad_method in ["detach", "undetach"]

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        print("***********ndisps:{}  disp_interval_pixel:{} grad_method:{} ns:{}  ns_size:{} cr_base_chs:{} ************".format(
            self.ndisps, self.disp_interval_pixel, self.grad_method, self.using_ns, self.ns_size, self.cr_base_chs))

        self.feature_extraction = feature_extraction(num_stage=self.num_stage, arch_mode="fpn")

        self.get_cv = GetCostVolume()

        cr_feats_in_chs = [chs * 2 for chs in self.feature_extraction.out_channels]
        self.cost_agg = nn.ModuleList([CostAggregation(in_channels=cr_feats_in_chs[i], base_channels=cr_base_chs[i])
                                       for i in range(self.num_stage)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right, min_depth=0.1, max_depth=8.0):

        refimg_msfea = self.feature_extraction(left)
        targetimg_msfea = self.feature_extraction(right)

        outputs = {}
        pred, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            if pred is not None:
                if self.grad_method == "detach":
                    cur_depth = pred.detach()
                else:
                    cur_depth = pred
            depth_range_samples, interval, cur_depth_min, cur_depth_max = get_depth_range_samples(cur_depth=cur_depth, 
                                                        ndepth=self.ndisps[stage_idx],
                                                        depth_inteval_pixel=self.disp_interval_pixel[stage_idx],
                                                        dtype=left.dtype,
                                                        device=left.device,
                                                        shape=[left.shape[0], left.shape[2], left.shape[3]],
                                                        min_depth=min_depth,
                                                        max_depth=max_depth,
                                                        using_ns=self.using_ns,
                                                        ns_size=self.ns_size)
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            refimg_fea, targetimg_fea = refimg_msfea["stage{}".format(stage_idx + 1)], \
                                        targetimg_msfea["stage{}".format(stage_idx + 1)]

            depth_rangel_samples_cv = F.interpolate((depth_range_samples//stage_scale).unsqueeze(1),
                        [self.ndisps[stage_idx]//int(stage_scale), left.size()[2]//int(stage_scale), left.size()[3]//int(stage_scale)],
                        mode='trilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            
                                                             
            # matching
            cost = self.get_cv(refimg_fea, targetimg_fea,
                               depth_range_samples=depth_rangel_samples_cv,
                               ndisp=self.ndisps[stage_idx]//int(stage_scale))
            #print(stage_idx, cost.shape)
            if self.training:
                pred0, pred1, pred2, pred3 = self.cost_agg[stage_idx](cost,
                                                                      FineD=self.ndisps[stage_idx],
                                                                      FineH=left.shape[2],
                                                                      FineW=left.shape[3],
                                                                      depth_range_samples=depth_range_samples,
                                                                      interval=interval)
                pred = pred3
                outputs_stage = {
                    "pred0": pred0,
                    "pred1": pred1,
                    "pred2": pred2,
                    "pred3": pred3,
                    "pred": pred}
                outputs["stage{}".format(stage_idx + 1)] = outputs_stage
                outputs.update(outputs_stage)

            else:
                pred3 = self.cost_agg[stage_idx](cost,
                                                 FineD=self.ndisps[stage_idx],
                                                 FineH=left.shape[2],
                                                 FineW=left.shape[3],
                                                 depth_range_samples=depth_range_samples,
                                                 interval=interval)
                pred = pred3
                outputs_stage = {
                    "pred3": pred3,
                    "pred": pred}
                outputs["stage{}".format(stage_idx + 1)] = outputs_stage

        return outputs
