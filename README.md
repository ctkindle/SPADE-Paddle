# SPADE-Paddle

使用Paddle2.1复现带SPADE正则化的像素风格迁移网络。

- [SPADE-Paddle](#spade-paddle)
  * [一、简介](#一简介)
  * [二、复现效果](#二复现效果)
  * [三、数据集](#三数据集)
  * [四、环境依赖](#四环境依赖)
  * [五、快速开始](#五快速开始)
    + [step1、clone](#step1clone)
    + [step2、训练](#step2训练)
    + [step3、验证](#step3验证)
    + [step4、预测](#step4预测)
  * [六、代码结构与详细说明](#六代码结构与详细说明)
    + [1、代码结构](#1代码结构)
    + [2、参数说明](#2参数说明)
    + [3、训练流程](#3训练流程)
    + [4、验证流程](#4验证流程)
    + [5、预测流程](#5预测流程)
    + [6、使用预训练模型预测](#6使用预训练模型预测)
      - [step1、下载预训练模型](#step1下载预训练模型)
      - [step2、预训练模型安装](#step2预训练模型安装)
      - [step3、使用预训练模型预测](#step3使用预训练模型预测)
  * [七、模型信息](#七模型信息)

## 一、简介

SPADE模型是 NVIDIA Opens AI Research Lab 出品的像素风格迁移网络，也被称为GauGAN，其使用 Spatially-Adaptive Normalization （SPADE）解决了Pix2PixHD模型的生成器在进行IN归一化时信息丢失的问题。GauGAN也采用了Pix2PixHD模型的 Coarse-to-fine generator（精炼生成器） 和 Multi-scale Discriminators（多尺度判别器），并使用VGG计算的 Perceptual Loss 增强生成图片的效果。

## 二、复现效果

下图为coco2017数据集验证集前10张图片（非刻意挑选）生成效果对比。左一列为复现效果，左二列为pytorch模型生成效果，左三列为数据集原图，右边三列为对应的deeplabv2模型（原论文采用：[https://github.com/kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) ）的语义分割图。验证集前100张图片（非刻意挑选）生成效果对比图存放在 result 文件夹下。

![](https://ai-studio-static-online.cdn.bcebos.com/6c07080c05c54761b933b143151a8dd8dd30f0a102bb4f8499ddfe86a4eef52e)

将两个模型生成的图片通过分割模型测试的数据如下：

|  | Pixel Accuracy | Frequency Weighted IOU |
| -------- | -------- | -------- |
| Pytorch 模型     | 0.71063813     | 0.62763964     |
| Paddle 模型    | 0.71857583     | 0.63593917     |

## 三、数据集

使用的数据集为COCO2017数据集，我已经使用原论文代码中的脚本将其处理成了适用于SPADE模型训练的版本，主要就是添加了instance图片标注。处理后的数据集已公开在AI Studio上：[https://aistudio.baidu.com/aistudio/datasetdetail/96023](https://aistudio.baidu.com/aistudio/datasetdetail/96023)

我也在百度网盘上存了一份数据集，方便大家下载。模型训练需要使用coco2017数据集中的“train2017.zip”、“val2017.zip”、“stuffthingmaps_trainval2017.zip” 三个文件，可在[此处](https://github.com/nightrome/cocostuff)下载。 其中，train2017.zip 中的图片文件解压复制到 dataset/coco_stuff/train_img 文件夹下，stuffthingmaps_trainval2017.zip 中的训练集语义分割标签 png 图片解压复制到 dataset/coco_stuff/train_label 文件夹下。额外需要的“实例+语义分割标签”，我已用脚本处理好，可在百度网盘 [https://pan.baidu.com/s/162Ogv-JzKSmzzWfP0iPW5A](https://pan.baidu.com/s/162Ogv-JzKSmzzWfP0iPW5A)上下载，提取码：“pdpd”。 这里的图片解压复制到 dataset/coco_stuff/train_inst 文件夹下。

+ 数据集大小：图片的语义标签共182个类别，123K图片
	+ 训练集：118K图片img、118K对应语义分割标签label、118K对应的实例分割标签inst
	+ 验证集：5K图片img、5K对应语义分割标签label、5K对应的实例分割标签inst
+ 数据格式：图片数据img为jpg格式，语义分割标签label和实例分割标签isnt为png格式

更多COCO2017数据集信息请参考：[https://github.com/nightrome/cocostuff](https://github.com/nightrome/cocostuff)

## 四、环境依赖
+ 硬件：GPU、CPU
+ 框架：
		+ PaddlePaddle >= 2.0.0

## 五、快速开始

（本项目的“包括额外预训练模型和数据标注的”一键运行版本已公开在 AI Studio 上：[https://aistudio.baidu.com/aistudio/clusterprojectdetail/2192011](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2192011)）

### step1、clone

```
# clone this repo
git clone https://github.com/ctkindle/SPADE-Paddle
cd SPADE-Paddle
```

### step2、训练


```
# 单卡训练
python train.py
```

输出


```
初始化训练轮数，开始第 1 轮训练...
epoch: 1 step: 0 g_loss_gan: [0.13950709] g_loss_feat: [5.1648903] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.93201095] d_loss_r: [1.1766295]
epoch: 1 step: 1 g_loss_gan: [0.10469157] g_loss_feat: [4.335578] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9642898] d_loss_r: [1.1653838]
epoch: 1 step: 2 g_loss_gan: [0.1455451] g_loss_feat: [6.699507] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91549283] d_loss_r: [1.1782851]
epoch: 1 step: 3 g_loss_gan: [0.14599623] g_loss_feat: [4.8984804] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91000307] d_loss_r: [1.1716485]
epoch: 1 step: 4 g_loss_gan: [0.14762324] g_loss_feat: [6.9864326] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9373741] d_loss_r: [1.1693909]
epoch: 1 step: 5 g_loss_gan: [0.13426134] g_loss_feat: [6.2136183] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92324626] d_loss_r: [1.1552694]
epoch: 1 step: 6 g_loss_gan: [0.13324602] g_loss_feat: [5.2056603] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92374295] d_loss_r: [1.1546884]
epoch: 1 step: 7 g_loss_gan: [0.13830245] g_loss_feat: [5.179484] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9370585] d_loss_r: [1.1726236]
epoch: 1 step: 8 g_loss_gan: [0.13696505] g_loss_feat: [6.920957] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92104435] d_loss_r: [1.1741142]
epoch: 1 step: 9 g_loss_gan: [0.10034883] g_loss_feat: [5.3364086] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92801106] d_loss_r: [1.1365886]
epoch: 1 step: 10 g_loss_gan: [0.13515502] g_loss_feat: [5.432964] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91358155] d_loss_r: [1.149337]
epoch: 1 step: 11 g_loss_gan: [0.13393757] g_loss_feat: [3.1982384] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91973245] d_loss_r: [1.1477559]
epoch: 1 step: 12 g_loss_gan: [0.13413814] g_loss_feat: [8.548164] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92463267] d_loss_r: [1.1617179]
第[1]轮模型保存完成。用时[00:11:40] 开始时间： 2021-09-15 15:02:49 结束时间： 2021-09-15 15:14:29
```

### step3、验证

```
python val.py
```

输出

```
已经完成 [1] 轮训练，开始验证...
读取存储的模型权重、优化器参数...
[0] 000000017914已存储至：output_val/pics
[1] 000000029286已存储至：output_val/pics
[2] 000000138805已存储至：output_val/pics
[3] 000000184101已存储至：output_val/pics
[4] 000000197384已存储至：output_val/pics
[5] 000000203744已存储至：output_val/pics
[6] 000000284465已存储至：output_val/pics
[7] 000000350505已存储至：output_val/pics
[8] 000000371376已存储至：output_val/pics
[9] 000000426773已存储至：output_val/pics
[10] 000000475177已存储至：output_val/pics
[11] 000000500044已存储至：output_val/pics
[12] 000000580986已存储至：output_val/pics
```

### step4、预测

```
python predict.py
```

输出

```
开始验证...
读取存储的模型权重...
预测完成，输出图片存储至：prediction/result.jpg
```

## 六、代码结构与详细说明

### 1、代码结构

│  predict.py			# 预测脚本

│  README.md

│  run.ipynb			# jupyter notebook 脚本

│  train.py				# 训练脚本

│  train_multigpu.sh	# 多卡训练脚本

│  val.py				# 预测脚本

│

├─config

│  └─init.py			# 全局参数存储脚本，包括训练、验证、预测用到的参数

│

├─dataset				# 数据集

│  └─coco_stuff

│      ├─train_img		# 训练集图片

│      ├─train_inst		# 训练集实例分割标签

│      ├─train_label	# 训练集语义分割标签

│      ├─val_img		# 验证集图片

│      ├─val_inst		# 验证集实例分割标签

│      └─val_label		# 验证集语义分割标签

│

├─model

│  │  data.py				# 数据处理脚本定义DataLoader

│  │  discriminator.py		# 判别器

│  │  generator.py			# 生成器

│  │  vae_encoder.py		# 提取图片特征的vae编码器

│  └─vgg19.py				# 用于计算 Perceptu Loss 的VGG19网络

│

├─output

│  │  current_epoch.npy		# 存储当前训练轮数

│  │  log.npy				# 训练日志

│  │

│  ├─model					# 存储的 checkpoint 模型

│  │  │  n_d.pdopt			# 判别器优化器参数

│  │  │  n_d.pdparams		# 判别器权重

│  │  │  n_g.pdopt			# 生成器优化器参数

│  │  └─n_g.pdparams		# 判别器权重

│  │

│  └─pics					# 训练过程生成的图片

│

├─output_val				# 验证过程生成的图片

│  └─pics

│

├─prediction				# 预测图片

│  │  result.jpg			# 预测结果

│  └─test.png				# 预测输入的语义标签

│

├─result					# 使用复现模型权重在验证集上生成的结果

│

└─utils

│  └─util.py				# 项目用到的其它数据处理、存储等辅助函数、类

### 2、参数说明

模型训练、验证、预测时用到的全局参数全部存放在 config/init.py 脚本里

| **参数** | **默认值** | **说明** | **其他** |
| - | - | - | - |
| aspect_ratio | 1.0 | 计算生成器潜变量大小时使用的图片宽高比例 |  |
| batchSize | 1 | batch尺寸 |  |
| beta1 | 0.0 | adam优化器超参 |  |
| beta2 | 0.999 | adam优化器超参 |  |
| contain_dontcare_label | True | 语义分割图是否包含未知类别 |  |
| crop_size | 256 | 训练图片裁切尺寸 |  |
| label_nc | 182 | 语义标签类别数 |  |
| lambda_feat | 10.0 | 判别器各个尺度特征权重比例 | 相邻两层之间 |
| load_size | 286 | 图片读取尺寸 |  |
| lr | 0.0001 | 学习率 |  |
| n_layers_D | 4 | 判别器深度 |  |
| ndf | 64 | 判别器宽度 |  |
| nef | 16 | VAE编码器宽度 |  |
| ngf | 64 | 生成器宽度 |  |
| no_TTUR | False | 生成器、判别器是否使用不同学习率 |  |
| no_instance | False | 不适用实例分割标签 |  |
| no_vgg_loss | False | 不适用 perceptual loss |  |
| norm_D | 'spectralinstance' | 判别器正则化方式 |  |
| norm_E | 'spectralinstance' | VAE正则化方式 |  |
| norm_G | 'spectralspadesyncbatch3x3' | 生成器正则化方式 | 单卡使用'spectralspadebatch3x3' |
| num_D | 2 | 判别器个数 |  |
| num_upsampling_layers | 'normal' | 生成器上采样层数 |  |
| output_nc | 3 | 输出通道数 | 3为rgb |
| semantic_nc | 184 | 语义标签数 | 182个类别+1未知类别+1实例分割标签 |
| use_vae | True | 使用VAE编码器 |  |
| z_dim | 256 | 输入生成器的噪声维度 |  |
| dataroot  |  'dataset/' | 数据集根目录 |  |
| datasetdir  |  '' | 数据集目录 | 脚本训练使用 |
| vggwpath  |  'vgg/vgg19pretrain.pdparams' | VGG预训练权重路径 |  |
| output  |  'output/' | 输出模型权重存储路径 | checkpoints |
| lastoutput  |  'output/' | 读取模型权重路径 | checkpoints |
| predict_inst  |  'prediction/test.png' | 预测实例分割标签存放路径 |  |
| predict_result  |  'prediction/result.jpg' | 预测结果图片存放路径 |  |



### 3、训练流程

#### 单机训练

```
python train.py
```

#### 多机训练

```
python -m paddle.distributed.launch train.py
```

训练输出

```
初始化训练轮数，开始第 1 轮训练...
epoch: 1 step: 0 g_loss_gan: [0.13950709] g_loss_feat: [5.1648903] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.93201095] d_loss_r: [1.1766295]
epoch: 1 step: 1 g_loss_gan: [0.10469157] g_loss_feat: [4.335578] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9642898] d_loss_r: [1.1653838]
epoch: 1 step: 2 g_loss_gan: [0.1455451] g_loss_feat: [6.699507] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91549283] d_loss_r: [1.1782851]
epoch: 1 step: 3 g_loss_gan: [0.14599623] g_loss_feat: [4.8984804] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91000307] d_loss_r: [1.1716485]
epoch: 1 step: 4 g_loss_gan: [0.14762324] g_loss_feat: [6.9864326] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9373741] d_loss_r: [1.1693909]
epoch: 1 step: 5 g_loss_gan: [0.13426134] g_loss_feat: [6.2136183] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92324626] d_loss_r: [1.1552694]
epoch: 1 step: 6 g_loss_gan: [0.13324602] g_loss_feat: [5.2056603] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92374295] d_loss_r: [1.1546884]
epoch: 1 step: 7 g_loss_gan: [0.13830245] g_loss_feat: [5.179484] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.9370585] d_loss_r: [1.1726236]
epoch: 1 step: 8 g_loss_gan: [0.13696505] g_loss_feat: [6.920957] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92104435] d_loss_r: [1.1741142]
epoch: 1 step: 9 g_loss_gan: [0.10034883] g_loss_feat: [5.3364086] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92801106] d_loss_r: [1.1365886]
epoch: 1 step: 10 g_loss_gan: [0.13515502] g_loss_feat: [5.432964] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91358155] d_loss_r: [1.149337]
epoch: 1 step: 11 g_loss_gan: [0.13393757] g_loss_feat: [3.1982384] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.91973245] d_loss_r: [1.1477559]
epoch: 1 step: 12 g_loss_gan: [0.13413814] g_loss_feat: [8.548164] g_loss_vgg: None g_loss_vae: None d_loss_f: [0.92463267] d_loss_r: [1.1617179]
第[1]轮模型保存完成。用时[00:11:40] 开始时间： 2021-09-15 15:02:49 结束时间： 2021-09-15 15:14:29
```

此时，程序会将主进程的输出log导入到 output/log.npy 文件中，是使用 numpy.save() 存储的，使用 numpy.load() 读取即可。每个step存储一条log，共465798条。每条 log 存储7个loss部分，从左到右分别是：
1. g_loss_gan：生成器 loss
1. g_loss_feat：判别器多尺度特征 loss
1. g_loss_vgg：辅助 vgg19 预训练模型的 perceptual loss
1. g_loss_vae：使用子编码器学习的 style loss
1. d_loss：判别器总 loss ，是后面两项的和
1. d_loss_f：判别器判别生成图片的 loss
1. d_loss_r：判别器判真实成图片的 loss

所有的全局参数、超参都写到了 model/init.py 脚本里的 OPT 类里，可使用默认值或根据需求修改。论文使用了通过 vgg19 预训练模型计算的 perceptual loss 增强生成效果，所以要启用 vgg loss，要先在 model/init.py 脚本的 OPT 类里设置 `self.no_vgg_loss=True`，然后将 vgg19 预训练模型的参数 vgg19pretrain.pdparams 复制到 vgg 文件夹下。权重可在百度网盘下载 [https://pan.baidu.com/s/1t5z_uAdNklnFrPfTYOUu9A](https://pan.baidu.com/s/1t5z_uAdNklnFrPfTYOUu9A) ，提取码 “pdpd”。

训练后的权重存储在 output/model 文件夹下，可用于继续训练和预测。

多卡训练应修改 model/init.py 脚本里 OPT 类的设置为：`self.norm_G='spectralspadesyncbatch3x3'`，以启用跨多 GPU 的 BN来增强生成效果。大BatchSize效果更好，BigGAN说的~~ 。

### 4、验证流程

```
python val.py
```

输出

```
已经完成 [1] 轮训练，开始验证...
读取存储的模型权重、优化器参数...
[0] 000000017914已存储至：output_val/pics
[1] 000000029286已存储至：output_val/pics
[2] 000000138805已存储至：output_val/pics
[3] 000000184101已存储至：output_val/pics
[4] 000000197384已存储至：output_val/pics
[5] 000000203744已存储至：output_val/pics
[6] 000000284465已存储至：output_val/pics
[7] 000000350505已存储至：output_val/pics
[8] 000000371376已存储至：output_val/pics
[9] 000000426773已存储至：output_val/pics
[10] 000000475177已存储至：output_val/pics
[11] 000000500044已存储至：output_val/pics
[12] 000000580986已存储至：output_val/pics
```

模型验证需要读训练时取存储的生成器模型权重或预训练权重文件：output/model/n_g.pdparams 。预测输出的图片存储在 output_val/pics 文件夹下。

### 5、预测流程

```
python predict.py
```

输出

```
开始验证...
读取存储的模型权重...
预测完成，输出图片存储至：prediction/result.jpg
```

输入、输出图片默认存放在 prediction 文件夹下，可通过修改init.py脚本中的参数设置进行修改。

### 6、使用预训练模型预测

#### step1、下载预训练模型

权重可在百度网盘下载 [https://pan.baidu.com/s/1p4Bpo2ymdA2C2VGp-wV3wA](https://pan.baidu.com/s/1p4Bpo2ymdA2C2VGp-wV3wA) ，提取码：“pdpd”。

#### step2、预训练模型安装

将下载的三个模型文件复制到 output/model 文件夹下即可。

#### step3、使用预训练模型预测

```
python val.py
```

输出

```
已经完成 [170] 轮训练，开始验证...
读取存储的模型权重、优化器参数...
[0] 000000017914已存储至：output_val/pics
[1] 000000029286已存储至：output_val/pics
[2] 000000138805已存储至：output_val/pics
[3] 000000184101已存储至：output_val/pics
[4] 000000197384已存储至：output_val/pics
[5] 000000203744已存储至：output_val/pics
[6] 000000284465已存储至：output_val/pics
[7] 000000350505已存储至：output_val/pics
[8] 000000371376已存储至：output_val/pics
[9] 000000426773已存储至：output_val/pics
[10] 000000475177已存储至：output_val/pics
[11] 000000500044已存储至：output_val/pics
[12] 000000580986已存储至：output_val/pics
```

## 七、模型信息

关于模型的其他信息，可以参考下表：

| **信息** | **说明** |
| - | - |
| 发布者 | FutureSI |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1 |
| 应用场景 | 图像生成 |
| 支持硬件 | GPU、CPU |
| DeepLabV2 Pixel Accuracy | 0.71857583 |
| 预训练权重 | [网盘链接](https://pan.baidu.com/s/1p4Bpo2ymdA2C2VGp-wV3wA)（提取码：pdpd） |
| VGG19预训练权重 | [网盘链接](https://pan.baidu.com/s/1t5z_uAdNklnFrPfTYOUu9A)（提取码：pdpd） |
| 模型源代码（1） | [生成器](https://github.com/ctkindle/SPADE-Paddle/blob/main/model/generator.py) |
| 模型源代码（2） | [判别器](https://github.com/ctkindle/SPADE-Paddle/blob/main/model/discriminator.py) |
| 模型源代码（3） | [VAE编码器](https://github.com/ctkindle/SPADE-Paddle/blob/main/model/vae_encoder.py) |
| 模型源代码（4） | [VGG19](https://github.com/ctkindle/SPADE-Paddle/blob/main/model/vgg19.py) |
| 在线运行 | [AI Studio项目地址](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2192011) |
