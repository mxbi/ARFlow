import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward
import torch

def cn(t):
    if torch.isnan(t).any():
        raise ValueError('nan found in tensor, shape={}, count={}'.format(t.shape, torch.isnan(t).sum()))

class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
           loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                     im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            flow_12 = flow[:, :2]
            flow_21 = flow[:, 2:]
            # backwards warp im2->im1 and im1->im2
            if True: # Use softsplat!
                import softsplat

                im1_warped = flow_warp(im2_scaled, flow_12, pad=self.cfg.warp_pad)
                im2_warped = flow_warp(im1_scaled, flow_21, pad=self.cfg.warp_pad)

                im1_metric = torch.nn.functional.l1_loss(im1_warped, im1_scaled, reduction='none').mean(dim=1, keepdim=True)
                im2_metric = torch.nn.functional.l1_loss(im2_warped, im2_scaled, reduction='none').mean(dim=1, keepdim=True)

                # print(flow_21.min(), flow_21.max())
                im1_recons = 1*softsplat.softsplat(tenIn=im2_scaled, tenFlow=flow_21, tenMetric=(-20*im2_metric).clip(-20.0, 20.0), strMode='soft')
                im2_recons = 1*softsplat.softsplat(tenIn=im1_scaled, tenFlow=flow_12, tenMetric=(-20*im1_metric).clip(-20.0, 20.0), strMode='soft')
                # print(im1_recons.shape)

                if i == 0:
                    if self.cfg.occ_from_back:
                        occu_mask1 = 1 - get_occu_mask_backward(flow_12, th=0.2)
                        occu_mask2 = 1 - get_occu_mask_backward(flow_21, th=0.2)
                    else:
                        occu_mask1 = 1 - get_occu_mask_bidirection(flow_12, flow_21)
                        occu_mask2 = 1 - get_occu_mask_bidirection(flow_21, flow_12)
                else:
                    occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                            (h, w), mode='nearest')
                    occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                            (h, w), mode='nearest')

                occu_mask1 *= 1 - ((im1_recons == 0).sum(axis=1) == 3).unsqueeze(1).float()
                occu_mask2 *= 1 - ((im2_recons == 0).sum(axis=1) == 3).unsqueeze(1).float()

                # im2_recons = im2_recons / 2
                # im1_recons = im1_recons / 2
                # import mlcrate as mlc
                # if i == 0:
                    # mlc.save([im1_warped, im2_warped, im1_metric, im2_metric, im1_recons, im2_recons, occu_mask1, occu_mask2, flow_12, flow_21], 'im2_debug.pkl')
            else:
                im1_recons = flow_warp(im2_scaled, flow_12, pad=self.cfg.warp_pad)
                im2_recons = flow_warp(im1_scaled, flow_21, pad=self.cfg.warp_pad)

                # Estimate an occlusion mask
                # This can be either done using backwards flow or bidirectionally
                # For later layers, we downscale the mask from the previous layer instead
                if i == 0:
                    if self.cfg.occ_from_back:
                        occu_mask1 = 1 - get_occu_mask_backward(flow_12, th=0.2)
                        occu_mask2 = 1 - get_occu_mask_backward(flow_21, th=0.2)
                    else:
                        occu_mask1 = 1 - get_occu_mask_bidirection(flow_12, flow_21)
                        occu_mask2 = 1 - get_occu_mask_bidirection(flow_21, flow_12)
                else:
                    occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                            (h, w), mode='nearest')
                    occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                            (h, w), mode='nearest')

                import mlcrate as mlc
                if i == 0:
                    mlc.save([im1_scaled, im2_scaled, im1_recons, im2_recons, occu_mask1, occu_mask2, flow_12, flow_21], 'im2_debug_old.pkl')

            cn(im1_recons)
            cn(im2_recons)

            cn(occu_mask1)
            cn(occu_mask2)

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow_12 / s, im1_scaled)

            if self.cfg.with_bk: # Bidirectional loss
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow_21 / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        total_loss = warp_loss + smooth_loss

        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()
