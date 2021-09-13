from config.init import OPT
from model.data import COCODateset, DataLoader
from model.generator import SPADEGenerator
from model.vae_encoder import VAE_Encoder
from model.discriminator import MultiscaleDiscriminator
from model.vgg19 import VGG19, center_crop
from utils.util import data_onehot_pro, save_pics

import numpy as np
import time
import os
from PIL import Image, ImageOps

import paddle
import paddle.nn as nn

opt = OPT()
opt.batchSize=1

def predict(opt):
    print('开始验证...')

    # 初始化模型
    G = SPADEGenerator(opt)
    G.eval()
    if opt.use_vae:
        E = VAE_Encoder(opt)
        E.train()

    # 读取保存的模型权重、优化器参数
    print('读取存储的模型权重...')

    g_statedict_model = paddle.load(os.path.join(opt.lastoutput, "model/n_g.pdparams"))
    G.set_state_dict(g_statedict_model)

    if opt.use_vae:
        e_statedict_model = paddle.load(os.path.join(opt.lastoutput, "model/n_e.pdparams"))
        E.set_state_dict(e_statedict_model)
        
    # 读取预测图片
    inst = Image.open(opt.predict_inst)
    inst = inst.resize((opt.crop_size, opt.crop_size), Image.NEAREST)
    inst = np.array(inst)
    for i in range(inst.shape[0]):
        for j in range(inst.shape[1]):
            if inst[i][j] >= opt.label_nc:
                inst[i][j] = opt.label_nc
    inst = inst.reshape((1, 1, opt.crop_size, opt.crop_size))
    inst = paddle.to_tensor(inst)
    
    if opt.use_vae:
        img = Image.open(opt.predict_img)
        img = img.resize((opt.crop_size, opt.crop_size), Image.BICUBIC)
        img = np.array(img)
        img = img.reshape((1, -1, opt.crop_size, opt.crop_size))
        img = paddle.to_tensor(img)
        mu, logvar = E(img)
        std = paddle.exp(0.5 * logvar)
        eps = paddle.to_tensor(np.random.normal(0, 1, (std.shape[0], std.shape[1])).astype('float32'))
        z = eps * std + mu

    # 开始预测
    one_hot = data_onehot_pro(inst, inst, opt)
    predicted = G(one_hot, z if opt.use_vae else None).numpy()[0]
    predicted = np.transpose(predicted, (1, 2, 0))
    predicted = ((predicted + 1.) / 2. * 255).astype('uint8')
    predicted = Image.fromarray(predicted)
    predicted.save(opt.predict_result)

predict(opt)
