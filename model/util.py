import paddle
import paddle.nn as nn
from paddle.nn import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, Linear
import paddle.nn.functional as F
import numpy as np
import math
import functools
from PIL import Image
import os

def simam(x, e_lambda=1e-4):
    b, c, h, w = x.shape
    n = w * h - 1
    x_minus_mu_square = (x - x.mean(axis=[2, 3], keepdim=True)) ** 2
    y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(axis=[2, 3], keepdim=True) / n + e_lambda)) + 0.5
    return x * nn.functional.sigmoid(y)

def save_pics(pics, file_name='tmp', save_path='./output/pics/'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(len(pics)):
        pics[i] = pics[i][0]
        if pics[i].shape[0] != 3:
            pics[i] = np.resize(pics[i], (3, pics[i].shape[1], pics[i].shape[2]))
        else:
            pics[i] = (pics[i] + 1.) / 2. * 255
    pic = np.concatenate(tuple(pics), axis=2)
    pic = pic.transpose((1,2,0))
    img = Image.fromarray(pic.astype('uint8')).convert('RGB')
    img.save(os.path.join(save_path, file_name+'.jpg'))

## data pre-process
def data_onehot_pro(instance, label, opt):
    shape = instance.shape
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc
    shape[1] = nc
    # one hot
    # print(label.shape, label.dtype, nc)
    semantics = paddle.nn.functional.one_hot(label.astype('int64'). \
        reshape([opt.batchSize, opt.crop_size, opt.crop_size]), nc). \
        transpose((0, 3, 1, 2))

    # edge
    edge = np.zeros(instance.shape, 'int64')
    t = instance.numpy()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge = paddle.to_tensor(edge).astype('float32')

    semantics = paddle.concat([semantics, edge], 1)
    return semantics

## normalization
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x

class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(
                                 n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # transpose dim to front
            weight_mat = weight_mat.transpose([self.dim] + [
                d for d in range(weight_mat.dim()) if d != self.dim
            ])

        height = weight_mat.shape[0]

        return weight_mat.reshape([height, -1])

    def compute_weight(self, layer, do_power_iteration):
        weight = getattr(layer, self.name + '_orig')
        u = getattr(layer, self.name + '_u')
        v = getattr(layer, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with paddle.no_grad():
                for _ in range(self.n_power_iterations):
                    v.set_value(
                        F.normalize(
                            paddle.matmul(
                                weight_mat,
                                u,
                                transpose_x=True,
                                transpose_y=False),
                            axis=0,
                            epsilon=self.eps, ))

                    u.set_value(
                        F.normalize(
                            paddle.matmul(weight_mat, v),
                            axis=0,
                            epsilon=self.eps, ))
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = paddle.dot(u, paddle.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, layer, inputs):
        setattr(
            layer,
            self.name,
            self.compute_weight(
                layer, do_power_iteration=layer.training))

    @staticmethod
    def apply(layer, name, n_power_iterations, dim, eps):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = layer._parameters[name]

        with paddle.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape

            # randomly initialize u and v
            u = layer.create_parameter([h])
            u = normal_(u, 0., 1.)
            v = layer.create_parameter([w])
            v = normal_(v, 0., 1.)
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)

        # delete fn.name form parameters, otherwise you can not set attribute
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + "_orig", weight)
        # still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an Parameter and
        # gets added as a parameter. Instead, we register weight * 1.0 as a plain
        # attribute.
        setattr(layer, fn.name, weight * 1.0)
        layer.register_buffer(fn.name + "_u", u)
        layer.register_buffer(fn.name + "_v", v)
        layer.register_forward_pre_hook(fn)
        return fn


def spectral_norm(layer,
                  name='weight',
                  n_power_iterations=1,
                  eps=1e-12,
                  dim=None):
    r"""
    This spectral_norm layer applies spectral normalization to a parameter according to the 
    following Calculation:
    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.
    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds.
    .. math::
        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}
        \mathbf{u} := \\frac{\mathbf{W} \mathbf{v}}{\|\mathbf{W} \mathbf{v}\|_2}
    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.
    .. math::
        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}
        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}
    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .
    Parameters:
        layer(Layer): Layer of paddle, which has weight.
        name(str, optional): Name of the weight parameter. Default: 'weight'.
        n_power_iterations(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        dim(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: None.
        
    Returns:
        The original layer with the spectral norm hook
    Examples:
       .. code-block:: python
            from paddle.nn import Conv2D
            from paddle.nn.utils import Spectralnorm
            conv = Conv2D(3, 1, 3)
            sn_conv = spectral_norm(conv)
            print(sn_conv)
            # Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)
            print(sn_conv.weight)
            # Tensor(shape=[1, 3, 3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [[[[-0.21090528,  0.18563725, -0.14127982],
            #           [-0.02310637,  0.03197737,  0.34353802],
            #           [-0.17117859,  0.33152047, -0.28408015]],
            # 
            #          [[-0.13336606, -0.01862637,  0.06959272],
            #           [-0.02236020, -0.27091628, -0.24532901],
            #           [ 0.27254242,  0.15516677,  0.09036587]],
            # 
            #          [[ 0.30169338, -0.28146112, -0.11768346],
            #           [-0.45765871, -0.12504843, -0.17482486],
            #           [-0.36866254, -0.19969313,  0.08783543]]]])
    """

    if dim is None:
        if isinstance(layer, (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose,
                              Linear)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(layer, name, n_power_iterations, dim, eps)
    return layer

def build_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we do not use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2D,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'syncbatch':
        norm_layer = functools.partial(
            nn.SyncBatchNorm,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer

## initialize layers
@paddle.no_grad()
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def uniform_(x, a=-1., b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def constant_(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)
    x.set_value(temp_value)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    # if hasattr(m, 'weight') and classname.find('Conv') != -1:
    #     normal_(m.weight, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     normal_(m.weight, 1.0, 0.02)
    #     constant_(m.bias, 0)
    
    # print('init', classname, hasattr(m, 'weight'))
    # for name,value in vars(m).items():
    #     print('*********%s=%s'%(name,value))
    if hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
        constant_(m.weight, 2e-2)
    if hasattr(m, 'bias') and classname.find('Conv') != -1:
        if m.bias is None == False: # 未禁用bias时设置
            constant_(m.bias, 0)

# 当使用谱归一化时，手动设置卷积层的初始值（由于参数名称的改变，weights_init无法正常工作）
# spn_conv_init_weight = nn.initializer.Constant(value=2e-2)
# spn_conv_init_bias = nn.initializer.Constant(value=.0)
spn_conv_init_weight = None
spn_conv_init_bias = None
