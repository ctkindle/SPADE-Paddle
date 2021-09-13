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

from config.init import OPT
from utils.util import build_norm_layer, spn_conv_init_weight, spn_conv_init_bias, spectral_norm

# 定义VAE编码器，提取图片风格
class VAE_Encoder(nn.Layer):
    def __init__(self, opt):
        super(VAE_Encoder, self).__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf

        InstanceNorm = build_norm_layer('instance')
        model = [
            spectral_norm(nn.Conv2D(3, ndf, kw, 2, pw,
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
            InstanceNorm(ndf),

            # nn.LeakyReLU(.2),
            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 1, ndf * 2, kw, 2, pw,
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
            InstanceNorm(ndf * 2),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 2, ndf * 4, kw, 2, pw,
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
            InstanceNorm(ndf * 4),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 4, ndf * 8, kw, 2, pw,
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
            InstanceNorm(ndf * 8),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 8, ndf * 8, kw, 2, pw,
                    weight_attr=spn_conv_init_weight,
                    bias_attr=spn_conv_init_bias)),
            InstanceNorm(ndf * 8),
        ]
        if opt.crop_size >= 256:
            model += [
                nn.GELU(),
                spectral_norm(nn.Conv2D(ndf * 8, ndf * 8, kw, 2, pw,
                        weight_attr=spn_conv_init_weight,
                        bias_attr=spn_conv_init_bias)),
                InstanceNorm(ndf * 8),
            ]
        model += [nn.GELU(),]

        self.flatten = nn.Flatten(1, -1)
        self.so = 4
        self.fc_mu = nn.Linear(ndf * 8 * self.so * self.so, 256)
        self.fc_var = nn.Linear(ndf * 8 * self.so * self.so, 256)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        
        x = self.flatten(x)

        return self.fc_mu(x), self.fc_var(x)

if __name__ == '__main__':
    opt = OPT()
    opt.batchSize = 1
    ve = VAE_Encoder(opt)
    x = paddle.ones([opt.batchSize, 3, opt.crop_size, opt.crop_size]) / 2.
    m, v = ve(x)
    print(m.shape, m)
    print(v.shape, v)


