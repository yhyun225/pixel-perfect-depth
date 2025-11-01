# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import torch


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "rmse_loss" == loss_name:
        criterion = RMSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif 'gradient_loss' == loss_name:
        criterion = GradientMatchingLoss()
    elif 'multi_scale_gradient_loss' == loss_name:
        criterion = MultiScaleGradientMatchingLoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion

# Scale shift invariant Gradient matching loss proposed in https://arxiv.org/pdf/1907.01341 (MIDAS)
# apply only on single scale
class GradientMatchingLoss:
    def __init__(self):
        pass
    
    def grad(self, R):
        """
        Args:
            R ('torch.Tensor'): difference between gt & prediction, 'R' term in MIDAS paper.
        """
        grad_x = R[..., :, 1:] - R[..., :, :-1]
        grad_y = R[..., 1:, :] - R[..., :-1, :]

        return grad_x, grad_y

    def __call__(self, depth_pred, depth_gt, mask):
        N = torch.sum(mask)     # number of valid pixels
        depth_diff = depth_pred - depth_gt
        grad_x, grad_y = self.grad(depth_diff)

        # vertical
        grad_x = torch.abs(grad_x)
        mask_x = mask[..., :, 1:] * mask[..., :, :-1]
        grad_x = grad_x * mask_x

        # horizontal
        grad_y = torch.abs(grad_y)
        mask_y = mask[..., 1:, :] * mask[..., :-1, :]
        grad_y = grad_y * mask_y

        # gradient loss
        loss = torch.sum(grad_x) + torch.sum(grad_y)
        loss = loss / N

        return loss

# Scale shift invariant Gradient matching loss proposed in https://arxiv.org/pdf/1907.01341 (MIDAS)
# apply only on multi scale
class MultiScaleGradientMatchingLoss:
    def __init__(self, k=4):
        self.k = k      # number of scales

# weighted multi-directional gradient loss, adopted from https://github.com/indu1ge/DepthMaster/blob/main/src/util/loss.py
# modified from the original implementation
class HuberLoss:
    def __init__(self, delta=0.2):
        self.delta = delta
        
    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        # Huber loss
        # Compute the difference between predicted and ground truth values
        diff = depth_gt - depth_pred
        
        # Compute absolute difference and squared difference
        abs_diff = torch.abs(diff)
        squared_diff = diff ** 2
        
        # Use conditional selection between L2 loss and L1 loss
        loss = torch.where(abs_diff > self.delta, 0.5 * squared_diff, self.delta * abs_diff - 0.5 * self.delta ** 2)
        
        # Return the mean loss over all valid samples
        if valid_mask is not None:
            return torch.mean(loss[valid_mask])
        else:
            return torch.mean(loss)

#  Root mean squared error loss
class RMSELoss:
    def __init__(self, eps=1e-8, **kwargs):
        self.eps = eps
        self.mse = torch.nn.MSELoss(**kwargs)
    
    def __call__(self, depth_pred, depth_gt):
        return torch.sqrt(self.mse(depth_pred, depth_gt) + self.eps)
        

class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss

## 