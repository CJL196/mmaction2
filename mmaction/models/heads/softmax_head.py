# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class SoftmaxHead(BaseHead):
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
                loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # print(x.shape)  # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)         # [N*num_segs, in_channels, 1, 1]
        # print(x.shape)
        x = x.view(x.size(0), -1)  # [N*num_segs, in_channels]
        # print(x.shape)
        cls_score = self.fc_cls(x) # [N*num_segs, cls]
        # print(cls_score.shape)
        cls_score = self.average_clip(cls_score,num_segs=num_segs)
        # print(cls_score.shape)
        # [N, num_classes]
        return cls_score
