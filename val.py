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

import paddle
import paddle.nn as nn

opt = OPT()
opt.batchSize=1
opt.output = 'output_val/'

def infer(opt, epoch_num=1, restart=False, show_interval=1, save_interval=1):
    last_output_path = opt.lastoutput
    current_epoch = np.load(os.path.join(last_output_path, 'current_epoch.npy'))[0]
    print('已经完成 ['+str(current_epoch)+'] 轮训练，开始验证...')
    log = np.load(os.path.join(last_output_path, 'log.npy'))

    # 建立输出文件夹
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    # 设置数据集
    cocods = COCODateset(opt)
    dataloader = DataLoader(cocods, shuffle=False, batch_size=opt.batchSize, drop_last=False, num_workers=0, use_shared_memory=False)

    # 初始化模型
    D = MultiscaleDiscriminator(opt)
    G = SPADEGenerator(opt)
    D.eval()
    G.eval()
    if opt.use_vae:
        E = VAE_Encoder(opt)
        E.train()

    # 读取保存的模型权重、优化器参数
    if not restart:
        print('读取存储的模型权重、优化器参数...')

        g_statedict_model = paddle.load(os.path.join(last_output_path, "model/n_g.pdparams"))
        G.set_state_dict(g_statedict_model)

        if opt.use_vae:
            e_statedict_model = paddle.load(os.path.join(last_output_path, "model/n_e.pdparams"))
            E.set_state_dict(e_statedict_model)
            
    # 开始用验证数据集进行验证
    for epoch in range(current_epoch + 1, current_epoch + epoch_num + 1):
        start = time.time()
        for step, data in enumerate(dataloader):
            image, inst, label, fname = data

            if opt.use_vae:
                mu, logvar = E(image.detach())
                std = paddle.exp(0.5 * logvar)
                eps = paddle.to_tensor(np.random.normal(0, 1, (std.shape[0], std.shape[1])).astype('float32'))
                z = eps * std + mu

            one_hot = data_onehot_pro(inst, label, opt)
            fake_img = G(one_hot, z if opt.use_vae else None)

            # 存储生成的图片
            if step % save_interval == 0:
                save_pics([fake_img, inst, image], file_name=fname[0].replace('.png', ''), save_path=os.path.join(opt.output, 'pics'))
                print('['+str(step)+']', fname[0].replace('.png', '')+'已存储至：'+os.path.join(opt.output, 'pics'))
                

infer(opt, epoch_num = 1, show_interval=400, restart=False)
