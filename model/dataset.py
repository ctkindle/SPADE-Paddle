import zipfile
import tarfile
import os
import shutil

from config.init import OPT

def build_dataset(dataroot, datasetdir, datadir='coco_stuff/'):
    # coco = zipfile.ZipFile(dataroot+datasetdir+'coco_stuff.zip')
    # coco.extractall(dataroot)
    # coco.close()

    if not os.path.exists(dataroot+datadir):
        os.mkdir(dataroot+datadir)

    img = zipfile.ZipFile(dataroot+datasetdir+'train2017.zip')
    img.extractall(dataroot+datadir)
    img.close()
    os.rename(dataroot+datadir+'train2017', dataroot+datadir+'train_img')
    for root, dirs, files in os.walk(dataroot+datadir+'train_img'):
        print('train_img:', len(files))

    lab = zipfile.ZipFile(dataroot+datasetdir+'stuffthingmaps_trainval2017.zip')
    lab.extractall(dataroot+datadir)
    lab.close()
    os.rename(dataroot+datadir+'train2017', dataroot+datadir+'train_label')
    shutil.rmtree(dataroot+datadir+'val2017')
    for root, dirs, files in os.walk(dataroot+datadir+'train_label'):
        print('train_label:', len(files))

    if not os.path.exists(dataroot+datadir+'train_inst'):
        os.mkdir(dataroot+datadir+'train_inst')
    ins = zipfile.ZipFile(dataroot+datasetdir+'train_inst.zip')
    ins.extractall(dataroot+datadir+'train_inst')
    ins.close()
    for root, dirs, files in os.walk(dataroot+datadir+'train_inst'):
        print('train_inst:', len(files))

def pro_checkpoint(lastoutput):
    tf = tarfile.TarFile(lastoutput)
    lo_path, _ = os.path.split(lastoutput)
    lo_path, _ = os.path.split(lo_path)
    tf.extractall(lo_path)
    names = tf.getnames()
    for name in names:
        p, n = os.path.split(name)
        if n == 'output':
            src = os.path.join(lo_path, name)
            # print(src)
    dst = os.path.join(lo_path, 'output_')
    # print(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(os.path.join(lo_path, 'output'))
    tf.close()
    os.rename(dst, os.path.join(lo_path, 'output'))

    print('解压的 checkpoint 文件：')
    for root, dirs, files in os.walk(os.path.join(lo_path, 'output')):
        print(files)
    

if __name__ == '__main__':
    opt = OPT()
    build_dataset(opt.dataroot, opt.datasetdir)
    if os.path.exists(opt.lastoutput):
        pro_checkpoint(opt.lastoutput)
    