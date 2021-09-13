#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

import numpy as np

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]

# VGG19 模型，已经预训练好，用于计算生成器的 perceptual loss
class ConvBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, groups, name=None):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name=name + "1_weights"),
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "2_weights"),
                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "3_weights"),
                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "4_weights"),
                bias_attr=False)

        self._pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            x = F.relu(x)
        x = self._pool(x)
        return x


class VGGNet(nn.Layer):
    def __init__(self, layers=11, class_dim=1000):
        super(VGGNet, self).__init__()

        self.layers = layers
        self.vgg_configure = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        self._conv_block_1 = ConvBlock(3, 64, self.groups[0], name="conv1_")
        self._conv_block_2 = ConvBlock(64, 128, self.groups[1], name="conv2_")
        self._conv_block_3 = ConvBlock(128, 256, self.groups[2], name="conv3_")
        self._conv_block_4 = ConvBlock(256, 512, self.groups[3], name="conv4_")
        self._conv_block_5 = ConvBlock(512, 512, self.groups[4], name="conv5_")

        self._drop = Dropout(p=0.5, mode="downscale_in_infer")
        self._fc1 = Linear(
            7 * 7 * 512,
            4096,
            weight_attr=ParamAttr(name="fc6_weights"),
            bias_attr=ParamAttr(name="fc6_offset"))
        self._fc2 = Linear(
            4096,
            4096,
            weight_attr=ParamAttr(name="fc7_weights"),
            bias_attr=ParamAttr(name="fc7_offset"))
        self._out = Linear(
            4096,
            class_dim,
            weight_attr=ParamAttr(name="fc8_weights"),
            bias_attr=ParamAttr(name="fc8_offset"))

    def forward(self, inputs):
        features = []
        features.append(inputs)
        x = self._conv_block_1(inputs)
        features.append(x)
        x = self._conv_block_2(x)
        features.append(x)
        x = self._conv_block_3(x)
        features.append(x)
        x = self._conv_block_4(x)
        features.append(x)
        x = self._conv_block_5(x)

        x = paddle.reshape(x, [0, -1])
        x = self._fc1(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._out(x)
        return x, features


def VGG11(**args):
    model = VGGNet(layers=11, **args)
    return model


def VGG13(**args):
    model = VGGNet(layers=13, **args)
    return model


def VGG16(**args):
    model = VGGNet(layers=16, **args)
    return model


def VGG19(**args):
    model = VGGNet(layers=19, **args)
    return model

# spade 模型输出尺寸是256*256，vgg 模型尺寸为224
def random_crop(img_tensor, opt, size=224):
    left = np.random.randint(0, opt.crop_size - size + 1)
    top = np.random.randint(0, opt.crop_size - size + 1)
    img_tensor = img_tensor[:, :, top:top+size, left:left+size]
    return img_tensor

def center_crop(img_tensor, opt, size=224):
    left_top = (opt.crop_size - size) // 2
    img_tensor = img_tensor[:, :, left_top:left_top+size, left_top:left_top+size]
    return img_tensor

if __name__ == '__main__':
    model = VGG19()
    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(model._conv_block_1.parameters())
