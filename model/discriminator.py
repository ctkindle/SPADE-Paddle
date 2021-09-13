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
import paddle.nn as nn
import numpy as np
import copy
from config.init import OPT
from utils.util import build_norm_layer, spn_conv_init_weight, spn_conv_init_bias, spectral_norm, simam

# 定义patch gan判别器
class NLayersDiscriminator(nn.Layer):
    def __init__(self, opt):
        super(NLayersDiscriminator, self).__init__()
        
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        layer_count = 0

        layer = nn.Sequential(
            nn.Conv2D(input_nc, nf, kw, 2, padw),
            nn.GELU()
        )
        self.add_sublayer('block_'+str(layer_count), layer)
        layer_count += 1

        feat_size_prev = np.floor((opt.crop_size + padw * 2 - (kw - 2)) / 2).astype('int64')
        InstanceNorm = build_norm_layer('instance')
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            feat_size = np.floor((feat_size_prev + padw * 2 - (kw - stride)) / stride).astype('int64')
            feat_size_prev = feat_size
            layer = nn.Sequential(
                spectral_norm(nn.Conv2D(nf_prev, nf, kw, stride, padw, 
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
                InstanceNorm(nf),
                nn.GELU()
            )
            self.add_sublayer('block_'+str(layer_count), layer)
            layer_count += 1

        layer = nn.Conv2D(nf, 1, kw, 1, padw)
        self.add_sublayer('block_'+str(layer_count), layer)
        layer_count += 1

    def forward(self, input):
        output = []
        for layer in self._sub_layers.values():
            output.append(simam(layer(input)))
            input = output[-1]

        return output

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

# 定义训练使用的多尺度判别器
class MultiscaleDiscriminator(nn.Layer):
    def __init__(self, opt):
        super(MultiscaleDiscriminator, self).__init__()
        
        for i in range(opt.num_D):
            sequence = []
            feat_size = opt.crop_size
            for j in range(i):
                sequence += [nn.AvgPool2D(3, 2, 1)]
                feat_size = np.floor((feat_size + 1 * 2 - (3 - 2)) / 2).astype('int64')
            opt_downsampled = copy.deepcopy(opt)
            opt_downsampled.crop_size = feat_size
            sequence += [NLayersDiscriminator(opt_downsampled)]
            sequence = nn.Sequential(*sequence)
            self.add_sublayer('nld_'+str(i), sequence)

    def forward(self, input):
        output = []
        for layer in self._sub_layers.values():
            output.append(layer(input))
        return output

if __name__ == '__main__':
    opt = OPT()
    opt.batchSize = 1
    md = MultiscaleDiscriminator(opt)
    np.random.seed(15)
    nld = NLayersDiscriminator(opt)
    input_nc = nld.compute_D_input_nc(opt)
    x = np.random.uniform(-1, 1, [opt.batchSize, input_nc, opt.crop_size, opt.crop_size]).astype('float32')
    x = paddle.to_tensor(x)
    y = md(x)
    for i in range(len(y)):
        for j in range(len(y[i])):
            print(i, j, y[i][j].shape)
        print('--------------------------------------')
    print(y)