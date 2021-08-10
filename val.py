from model.init import OPT
from model.data import COCODateset, DataLoader
from model.generator import SPADEGenerator
from model.vae_encoder import VAE_Encoder
from model.discriminator import MultiscaleDiscriminator
from model.vgg19 import VGG19, center_crop
from model.util import data_onehot_pro, save_pics

import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
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
#     if not opt.no_vgg_loss:
#         vgg19 = VGG19()
#         vgg_state_dict = paddle.load(opt.vggwpath)
#         vgg19.set_state_dict(vgg_state_dict)
#         l1loss = nn.loss.L1Loss()

#     # 设置优化器、学习率
#     if opt.no_TTUR:
#         opt_d = paddle.optimizer.Adam(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2, parameters=D.parameters())
#         if opt.use_vae:
#             opt_g = paddle.optimizer.Adam(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2, parameters=G.parameters() + E.parameters())
#         else:
#             opt_g = paddle.optimizer.Adam(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2, parameters=G.parameters())
#     else:
#         opt_d = paddle.optimizer.Adam(learning_rate=opt.lr * 2., beta1=opt.beta1, beta2=opt.beta2, parameters=D.parameters())
#         if opt.use_vae:
#             opt_g = paddle.optimizer.Adam(learning_rate=opt.lr / 2., beta1=opt.beta1, beta2=opt.beta2, parameters=G.parameters() + E.parameters())
#         else:
#             opt_g = paddle.optimizer.Adam(learning_rate=opt.lr / 2., beta1=opt.beta1, beta2=opt.beta2, parameters=G.parameters())

    # 读取保存的模型权重、优化器参数
    if not restart:
        print('读取存储的模型权重、优化器参数...')
#         d_statedict_model = paddle.load(os.path.join(last_output_path, "model/n_d.pdparams"))
#         D.set_state_dict(d_statedict_model)

        g_statedict_model = paddle.load(os.path.join(last_output_path, "model/n_g.pdparams"))
        G.set_state_dict(g_statedict_model)

        if opt.use_vae:
            e_statedict_model = paddle.load(os.path.join(last_output_path, "model/n_e.pdparams"))
            E.set_state_dict(e_statedict_model)

    for epoch in range(current_epoch + 1, current_epoch + epoch_num + 1):
        start = time.time()
        for step, data in enumerate(dataloader):
            if step >= 1:
                break
            
            image, inst, label, fname = data

            # train G
            if opt.use_vae:
                mu, logvar = E(image.detach())
                std = paddle.exp(0.5 * logvar)
                eps = paddle.to_tensor(np.random.normal(0, 1, (std.shape[0], std.shape[1])).astype('float32'))
                z = eps * std + mu
#                 g_vaeloss = -0.5 * paddle.sum(1. + logvar - paddle.pow(mu, 2) - paddle.exp(logvar))
#                 g_vaeloss *= opt.lambda_kld

            one_hot = data_onehot_pro(inst, label, opt)
            fake_img = G(one_hot, z if opt.use_vae else None)
#             fake_data = paddle.concat((one_hot, fake_img), 1)
#             real_data = paddle.concat((one_hot, image), 1)
#             fake_and_real_data = paddle.concat((fake_data, real_data), 0)
#             pred = D(fake_and_real_data)
            
#             g_ganloss = 0.
#             for i in range(len(pred)):
#                 pred_i = pred[i][-1][:opt.batchSize]
#                 new_loss = -pred_i.mean() # hinge loss
#                 g_ganloss += new_loss
#             g_ganloss /= len(pred)

#             g_featloss = 0.
#             for i in range(len(pred)):
#                 for j in range(len(pred[i]) - 1): # 除去最后一层的中间层featuremap
#                     unweighted_loss = (pred[i][j][:opt.batchSize] - pred[i][j][opt.batchSize:]).abs().mean() # L1 loss
#                     g_featloss += unweighted_loss * opt.lambda_feat / len(pred)

#             g_vggloss = paddle.to_tensor(0.)
#             if not opt.no_vgg_loss:
#                 # rates = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#                 _, fake_features = vgg19(center_crop(fake_img, opt, 224))
#                 _, real_features = vgg19(center_crop(image, opt, 224))
#                 for i in range(len(fake_features)):
#                     # g_vggloss += rates[i] * l1loss(fake_features[i], real_features[i])
#                     g_vggloss += l1loss(fake_features[i], real_features[i]) # 此vgg19预训练模型无bn层，所以尝试不用rate
#                 g_vggloss *= opt.lambda_vgg
            
#             if opt.use_vae:
#                 g_loss = g_ganloss + g_featloss + g_vggloss + g_vaeloss
#             else:
#                 g_loss = g_ganloss + g_featloss + g_vggloss
            
#             # train D
#             if opt.use_vae:
#                 # mu, logvar = E(paddle.ones([opt.batchSize, 3, opt.crop_size, opt.crop_size]))
#                 mu, logvar = E(image.detach())
#                 std = paddle.exp(0.5 * logvar)
#                 eps = paddle.to_tensor(np.random.normal(0, 1, (std.shape[0], std.shape[1])).astype('float32'))
#                 # eps = paddle.ones([std.shape[0], std.shape[1]]) / 278000.
#                 z = eps * std + mu

#             fake_img = G(one_hot, z if opt.use_vae else None)
#             fake_data = paddle.concat((one_hot, fake_img), 1)
#             fake_and_real_data = paddle.concat((fake_data, real_data), 0)
#             pred = D(fake_and_real_data)

#             df_ganloss = 0.
#             for i in range(len(pred)):
#                 pred_i = pred[i][-1][:opt.batchSize]
#                 new_loss = -paddle.minimum(-pred_i - 1, paddle.zeros_like(pred_i)).mean() # hingle loss
#                 df_ganloss += new_loss
#             df_ganloss /= len(pred)
            
#             dr_ganloss = 0.
#             for i in range(len(pred)):
#                 pred_i = pred[i][-1][opt.batchSize:]
#                 new_loss = -paddle.minimum(pred_i - 1, paddle.zeros_like(pred_i)).mean() # hingle loss
#                 dr_ganloss += new_loss
#             dr_ganloss /= len(pred)

#             d_loss = df_ganloss + dr_ganloss

            # save img
            if step % save_interval == 0:
                # print(fname[0])
                save_pics([fake_img, inst, image], file_name=fname[0].replace('.png', ''), save_path=os.path.join(opt.output, 'pics'))
                
#             # show log
#             if step % show_interval == 0:
#                 print('epoch:', epoch, 'step:', step, 'g_loss_gan:', g_ganloss.numpy(), 'g_loss_feat:', g_featloss.numpy(), \
#                     'g_loss_vgg:', g_vggloss.numpy() if not opt.no_vgg_loss else 'None', \
#                     'g_loss_vae:', g_vaeloss.numpy() if opt.use_vae else 'None', \
#                     'd_loss_f:', df_ganloss.numpy(), 'd_loss_r:', dr_ganloss.numpy())

#             # 写log
#             log_current_step = np.array([[
#                 g_ganloss.numpy()[0], \
#                 g_featloss.numpy()[0], \
#                 g_vggloss.numpy()[0] if not opt.no_vgg_loss else 0., \
#                 g_vaeloss.numpy()[0] if opt.use_vae else 0., \
#                 d_loss.numpy()[0], \
#                 df_ganloss.numpy()[0], \
#                 dr_ganloss.numpy()[0] \
#             ]])
#             log = np.concatenate((log, log_current_step), axis=0)

            # if step >= 100:
            #     break
        
#         # 存储模型
#         if epoch % 1 == 0:
#             end = time.time()
#             seconds = end - start
#             m, s = divmod(seconds, 60)
#             h, m = divmod(m, 60)
#             start_fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
#             end_fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
#             print('第['+str(epoch)+']轮模型保存完成。用时[%02d:%02d:%02d]' % (h, m, s), 
#                 '开始时间：', start_fmt, '结束时间：', end_fmt)

# opt.use_vae = False
infer(opt, epoch_num = 1, show_interval=400, restart=False)
