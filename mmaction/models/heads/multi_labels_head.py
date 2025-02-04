
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn
import torch
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead
import torch.nn.init as init
import torch.nn.functional as F


# @MODELS.register_module()
# class SigmoidHead(BaseHead):
#     """Class head for softmax.

#     Args:
#         num_classes (int): Number of classes to be classified.
#         in_channels (int): Number of channels in input feature.
#         loss_cls (dict or ConfigDict): Config for building loss.
#             Default: dict(type='CrossEntropyLoss').
#         spatial_type (str or ConfigDict): Pooling type in spatial dimension.
#             Default: 'avg'.
#         consensus (dict): Consensus config dict.
#         dropout_ratio (float): Probability of dropout layer. Default: 0.4.
#         init_std (float): Std value for Initiation. Default: 0.01.
#         kwargs (dict, optional): Any keyword argument to be used to initialize
#             the head.
#     """

#     def __init__(self,
#                 num_classes: int,
#                 in_channels: int,
#                 loss_cls: ConfigType = dict(type='BCELossWithLogits'),
#                 init_std: float = 0.01,
#                  **kwargs) -> None:
#         super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

#         self.init_std = init_std
#         self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

#     def init_weights(self) -> None:
#         """Initiate the parameters from scratch."""
#         normal_init(self.fc_cls, std=self.init_std)

#     def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
#         """Defines the computation performed at every call.

#         Args:
#             x (Tensor): The input data.
#             num_segs (int): Number of segments into which a video
#                 is divided.
#         Returns:
#             Tensor: The classification scores for input samples.
#         """
#         # print(x.shape)  # [N * num_segs, in_channels, 7, 7]
#         x = self.avg_pool(x)         # [N*num_segs, in_channels, 1, 1]
#         # print(x.shape)
#         x = x.view(x.size(0), -1)  # [N*num_segs, in_channels]
#         # print(x.shape)
#         cls_score = self.fc_cls(x) # [N*num_segs, cls]
#         # print(cls_score.shape)
#         cls_score = self.average_clip(cls_score,num_segs=num_segs)
#         # print(cls_score.shape)
#         # [N, num_classes]
#         return cls_score

class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()
    
    def forward(self, x, target):
        return (-target*(torch.log(x)) - (1-target)*torch.log(1-x)).mean(axis=1)

@MODELS.register_module()
class SigmoidHead(BaseHead):
    """Class head for softmax.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str or ConfigDict): Pooling type in spatial dimension.
            Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """


    def __init__(self,
                num_classes: int,
                in_channels: int,
                init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, **kwargs)
        pos_weight = torch.tensor([3.296, 5.03, 3.35, 12.24, 11.94], device='cuda') # ???
        self.init_std = init_std
        self.fc_layers = nn.ModuleList([nn.Linear(self.in_channels, 1) for _ in range(self.num_classes)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # BCE loss function
        self.loss_fn = loss_fn()

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for fc in self.fc_layers:
            # 使用 Xavier 初始化（正态分布）
            init.xavier_normal_(fc.weight)  # 使用 Xavier Normal 初始化权重
            if fc.bias is not None:
                init.zeros_(fc.bias)  # 初始化偏置为零

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # print('in forward')
        # print(x.shape)  # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)         # [N*num_segs, in_channels, 1, 1]
        # print(x.shape)
        x = x.view(x.size(0), -1)  # [N*num_segs, in_channels]
        # print(x.shape)
        logits = torch.cat([fc(x) for fc in self.fc_layers], dim=1)
        # print(logits)
        # print(logits.shape)
        # print('leave forward')
        logits = self.average_clip(logits,num_segs=num_segs)
        return logits


    
    def loss(self, feats, data_samples,num_segs,loss_aux=None):
        # print(f"from loss func: feats={feats.shape}, data_samples={len(data_samples)}, num_segs={num_segs}")
        logits = self(feats,num_segs)  # Get the logits for each sample
        logits = F.sigmoid(logits)  
        # print(f"logits: {logits}")
        # print(f"logits shape: {logits.shape}")
        labels = torch.stack([x.gt_label for x in data_samples])  # Stack labels for each sample
        labels = labels.float()  # Convert to float for BCE loss
        # print(labels)
        # BCE loss requires the shape of logits and labels to match
        loss_cls = self.loss_fn(logits, labels)
        # print(loss_cls)
        return dict(loss_cls=loss_cls)