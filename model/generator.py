import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import re

from config.init import OPT
from model.util import build_norm_layer, spn_conv_init_weight, spn_conv_init_bias, spectral_norm, simam

class SPADE(nn.Layer):
    def __init__(self, config_text, norm_nc, label_nc):
        super(SPADE, self).__init__()

        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        # print(config_text, parsed, param_free_norm_type, ks)

        self.param_free_norm = build_norm_layer(param_free_norm_type)(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(*[
            nn.Conv2D(label_nc, nhidden, ks, 1, pw),
            # nn.ReLU(),
            nn.GELU(),
        ])
        self.mlp_gamma = nn.Conv2D(nhidden, norm_nc, ks, 1, pw)
        self.mlp_beta = nn.Conv2D(nhidden, norm_nc, ks, 1, pw)

    def forward(self, x, segmap):
        # self.apply(weights_init)
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, x.shape[2:])
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        # return out
        return simam(out)

class SPADEResnetBlock(nn.Layer):
    def __init__(self, fin, fout, opt):
        super(SPADEResnetBlock, self).__init__()

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        # define spade layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.spade_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.spade_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.spade_s = SPADE(spade_config_str, fin, opt.semantic_nc)

        # define act_conv layers
        # SpectralNorm = build_norm_layer('spectral')
        self.act_conv_0 = nn.Sequential(*[
            # nn.LeakyReLU(.2),
            nn.GELU(),
            spectral_norm(nn.Conv2D(fin, fmiddle, 3, 1, 1, 
                weight_attr=spn_conv_init_weight,
                bias_attr=spn_conv_init_bias)),
            ])
        self.act_conv_1 = nn.Sequential(*[
            # nn.LeakyReLU(.2),
            nn.GELU(),
            spectral_norm(nn.Conv2D(fmiddle, fout, 3, 1, 1, 
                weight_attr=spn_conv_init_weight,
                bias_attr=spn_conv_init_bias)),
            ])
        if self.learned_shortcut:
            self.act_conv_s = nn.Sequential(*[
                spectral_norm(nn.Conv2D(fin, fout, 1, 1, 0, bias_attr=False,
                    weight_attr=spn_conv_init_weight)),
                ])


    def forward(self, x, seg):
        # self.apply(weights_init)

        x_s = self.shortcut(x, seg)

        dx = self.act_conv_0(self.spade_0(x, seg))
        dx = self.act_conv_1(self.spade_1(dx, seg))

        # return dx + x_s
        return simam(dx + x_s)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.act_conv_s(self.spade_s(x, seg))
        else:
            x_s = x
        return x_s

class SPADEGenerator(nn.Layer):
    def __init__(self, opt):
        super(SPADEGenerator, self).__init__()

        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if self.opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2D(self.opt.semantic_nc, 16 * nf, 3, 1, 1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2D(final_nc, 3, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, z=None):
        # self.apply(weights_init)
        # print(self.fc.parameters())
        seg = input
        if self.opt.use_vae:
            x = self.fc(z)
            # print(x)
            x = paddle.reshape(x, [-1, 16 * self.opt.ngf, self.sh, self.sw])
            # print(x)
        else:
            x = F.interpolate(seg, (self.sh, self.sw))
            x = self.fc(x)
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, .2))
        x = F.tanh(x)

        return x

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

if __name__ == '__main__':    
    opt = OPT()
    opt.batchSize = 1
    sg = SPADEGenerator(opt)
    np.random.seed(15)
    x = np.random.uniform(-1, 1, [opt.batchSize, opt.semantic_nc, opt.crop_size, opt.crop_size]).astype('float32')
    x = paddle.to_tensor(x)
    y = sg(x)
    print(y.shape)
    print(y)

