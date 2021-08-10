from paddle.io import Dataset, DataLoader
import os
import numpy as np
from PIL import Image, ImageOps
import random
import matplotlib.pyplot as plt

from model.init import OPT

# 处理图片数据：裁切、水平翻转、调整图片数据形状、归一化数据
def data_transform(img, resize_w, resize_h, load_size=286, pos=[0, 0, 256, 256], flip=True, is_image=True):
    if is_image:
        resized = img.resize((resize_w, resize_h), Image.BICUBIC)
    else:
        resized = img.resize((resize_w, resize_h), Image.NEAREST)
    croped = resized.crop((pos[0], pos[1], pos[2], pos[3]))
    # if is_image:
    #     croped = img.resize((256, 256), Image.BICUBIC)
    # else:
    #     croped = img.resize((256, 256), Image.NEAREST)
    fliped = ImageOps.mirror(croped) if flip else croped
    fliped = np.array(fliped) # transform to numpy array
    expanded = np.expand_dims(fliped, 2) if len(fliped.shape) < 3 else fliped
    transposed = np.transpose(expanded, (2, 0, 1)).astype('float32')
    if is_image:
        normalized = transposed / 255. * 2. - 1.
    else:
        normalized = transposed
    return normalized

# 定义CoCo数据集对象
class COCODateset(Dataset):
    def __init__(self, opt):
        super(COCODateset, self).__init__()
        # img_dir = opt.dataroot+'train_img/'
        # _, _, image_list = next(os.walk(img_dir))
        # self.image_list = np.sort(image_list)
        inst_dir = opt.dataroot+'coco_stuff/train_inst/'
        _, _, inst_list = next(os.walk(inst_dir))
        self.inst_list = np.sort(inst_list)
        self.opt = opt

    def __getitem__(self, idx):
        ins = Image.open(self.opt.dataroot+'coco_stuff/train_inst/'+self.inst_list[idx])
        lab = Image.open(self.opt.dataroot+'coco_stuff/train_label/'+self.inst_list[idx])
        img = Image.open(self.opt.dataroot+'coco_stuff/train_img/'+self.inst_list[idx].replace(".png", ".jpg"))
        img = img.convert('RGB')

        w, h = img.size
        resize_w, resize_h = 0, 0
        if w < h:
            resize_w, resize_h = self.opt.load_size, int(h * self.opt.load_size / w)
        else:
            resize_w, resize_h = int(w * self.opt.load_size / h), self.opt.load_size
        left = random.randint(0, resize_w - self.opt.crop_size)
        top = random.randint(0, resize_h - self.opt.crop_size)
        # top = random.randint(0, self.opt.load_size - self.opt.crop_size)
        # left = random.randint(0, self.opt.load_size - self.opt.crop_size)
        flip = True if random.randint(0, 100) > 50 else False
        
        img = data_transform(img, resize_w, resize_h, load_size=self.opt.load_size, 
            pos=[left, top, left + self.opt.crop_size, top + self.opt.crop_size], flip=flip, is_image=True)
        ins = data_transform(ins, resize_w, resize_h, load_size=self.opt.load_size, 
            pos=[left, top, left + self.opt.crop_size, top + self.opt.crop_size], flip=flip, is_image=False)
        lab = data_transform(lab, resize_w, resize_h, load_size=self.opt.load_size, 
            pos=[left, top, left + self.opt.crop_size, top + self.opt.crop_size], flip=flip, is_image=False)

        # 将label中的背景类别从255改为182
        mask = lab == 255
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        lab[mask] = nc - 1
        
        return img, ins, lab, self.inst_list[idx]

    def __len__(self):
        return len(self.inst_list)

if __name__ == '__main__':
    opt = OPT()
    opt.dataroot = '/home/aistudio/data/data96023/'
    # 定义图片loader
    cocods = COCODateset(opt)
    loader = DataLoader(cocods, shuffle=True, batch_size=1, drop_last=False, num_workers=1, use_shared_memory=False)
    for i, data in enumerate(loader):
        if i > 3 - 1:
            break
        img, ins, lab = data
        print('data shape:', img.shape, ins.shape, lab.shape)

        img = (np.transpose(img.numpy()[0], (1, 2, 0)) + 1.) / 2.
        ins = np.transpose(ins.numpy()[0], (1, 2, 0))[:, :, 0]
        lab = np.transpose(lab.numpy()[0], (1, 2, 0))[:, :, 0]

        plt.figure(figsize=(12,4),dpi=80)
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(ins)
        plt.subplot(1, 3, 3)
        plt.imshow(lab)
        plt.show()
