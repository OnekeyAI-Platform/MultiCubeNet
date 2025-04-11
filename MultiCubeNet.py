# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/12/08
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import deprecated_arg
from monai.utils.module import look_up_option

from onekey_algo.utils.about_log import logger
from onekey_core.core.losses_factory import cox_loss, create_losses


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


def get_conv1(conv1_t_size: int, conv1_t_stride: int):
    return (
        [0, conv1_t_size, (conv1_t_size, 7), (conv1_t_size, 7, 7)],
        [0, conv1_t_stride, (conv1_t_stride, 2), (conv1_t_stride, 2, 2)],
        [0, (conv1_t_size // 2), (conv1_t_size // 2, 3), (conv1_t_size // 2, 3, 3)],
    )


class MultiCubeNet(nn.Module):
    """
    Multi task, multi scale and multi layer 3D classification Network.

    """

    @deprecated_arg("n_classes", since="0.6")
    def __init__(
            self,
            output_channels: List[int],
            spatial_dims: int = 3,
            in_channels: int = 3,
            conv1_t_size: int = 7,
            conv1_t_stride: int = 1,
            no_max_pool: bool = False,
            widen_factor: float = 1.0,
            num_clf_classes: Union[int, List[int]] = 2,
            num_reg_tasks: Union[int] = None,
            feed_forward: bool = True,
            **kwargs
    ) -> None:
        super(MultiCubeNet, self).__init__()
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        # pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        self.downsample_layer = self.make_downsample_layer(2)
        self.layers = []
        layers = len(output_channels)
        self.params = []
        for i in range(layers):
            logger.info(f'Building layer {i} ...')
            former_channels = 0 if i == 0 else output_channels[i - 1]
            self.layers.append(self.make_layer(in_channels, blocks=layers - i, spatial_dims=spatial_dims,
                                               former_channels=former_channels, output_channels=output_channels[i:]))
        # Classification Task
        if isinstance(num_clf_classes, (list, tuple)):
            self.clf_tasks = [nn.Linear(output_channels[-1] * layers, nc) for nc in num_clf_classes]
        elif isinstance(num_clf_classes, int):
            self.clf_tasks = [nn.Linear(output_channels[-1] * layers, num_clf_classes)]
        else:
            self.clf_tasks = None

        # Regression Task
        if isinstance(num_reg_tasks, int):
            self.reg_tasks = [nn.Sequential(nn.Linear(output_channels[-1] * layers, 1, ), )
                              for _ in range(num_reg_tasks)]
        else:
            self.reg_tasks = None

        self.l1f, self.l1n = self.layers[0]
        self.l2f, self.l2n = self.layers[1]
        self.l3f, self.l3n = self.layers[2]
        self.l4f, self.l4n = self.layers[3]
        self.l5f, self.l5n = self.layers[4]
        self.clf_t1 = self.clf_tasks[0]
        self.clf_t2 = self.clf_tasks[1]
        self.clf_t3 = self.clf_tasks[2]
        self.reg_t = self.reg_tasks[0]

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    @staticmethod
    def make_downsample_layer(stride: int, spatial_dims: int = 3) -> nn.AdaptiveAvgPool3d:
        out = get_pool_layer(("avg", {"kernel_size": stride, "stride": stride}), spatial_dims=spatial_dims)
        # zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        # out = torch.cat([out.data, zero_pads], dim=1)
        return out

    @staticmethod
    def make_layer(
            in_channels: int,
            blocks: int,
            former_channels: int = 0,
            spatial_dims: int = 3,
            output_channels: List[int] = None,
    ) -> (nn.Sequential, nn.Sequential):

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]
        block_avgpool = get_avgpool()
        avgpool = avgp_type(block_avgpool[spatial_dims])
        layers = []
        assert len(output_channels) == blocks, f"number of output_channels not equal to blocks."
        start_in_channels = in_channels + former_channels
        for lidx, oc in enumerate(output_channels):
            logger.info(f'\tM: {lidx}, start_in_channels: {start_in_channels}, output_channels: {oc}')
            conv_layer = nn.Sequential(
                conv_type(start_in_channels, oc, kernel_size=3, stride=2, padding=1, bias=False, ),
                norm_type(oc),
                nn.ReLU(inplace=True))
            start_in_channels = oc
            layers.append(conv_layer)
        layers.append(avgpool)
        return layers[0], nn.Sequential(*layers[1:])

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        layer1_f, layer1_n = self.layers[0]
        former_out = layer1_f(x)
        outs = [layer1_n(former_out)]
        for layer_f, layer_n in self.layers[1:]:
            x = self.downsample_layer(x)
            combined_x = torch.concat([x, former_out], dim=1)
            former_out = layer_f(combined_x)
            outs.append(layer_n(former_out))
        outs = torch.concat(outs, dim=1).view(x.size(0), -1)
        outputs = []
        if self.clf_tasks is not None:
            outputs += [t(outs) for t in self.clf_tasks]
        if self.reg_tasks is not None:
            outputs += [t(outs) for t in self.reg_tasks]
        return outputs


if __name__ == '__main__':
    indata = torch.rand([2, 3, 96, 96, 96])
    t1_gt = torch.tensor([1, 1])
    t2_gt = torch.tensor([1, 0])
    t3_gt = torch.tensor([[355, 1], [3234, 0]])
    model = MultiCubeNet([256, 128, 64, 32, 16], num_clf_classes=[2, 2, 2], num_reg_tasks=1)
    t1_pred, t2_pred, t3_pred, t4_pred = model(indata)
    clf_loss = create_losses('softmax_ce')
    loss = cox_loss(t3_pred, t3_gt) + clf_loss(t1_pred, t1_gt) + clf_loss(t2_pred, t2_gt)
    print(model)
    # for param in model.params:
    #     print(list(param))
