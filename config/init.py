# 设置全局训练参数、验证参数、预测参数
class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        self.aspect_ratio=1.0
        self.batchSize=1
        self.beta1=0.0
        self.beta2=0.999
        self.contain_dontcare_label=True
        self.crop_size=256
        self.label_nc=182
        self.lambda_feat=10.0
        self.load_size=286
        self.lr=0.0001
        self.n_layers_D=4
        self.ndf=1 # 64
        self.nef=16
        self.ngf=1 # 64
        self.no_TTUR=False
        self.no_instance=False
        self.no_vgg_loss=True # False
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
        self.norm_G='spectralspadebatch3x3' # 多卡训练使用 'spectralspadesyncbatch3x3'
        self.num_D=2
        self.num_upsampling_layers='normal'
        self.output_nc=3
        self.semantic_nc=184
        self.use_vae=False # True
        self.z_dim=256
        self.dataroot = 'dataset/'
        self.datasetdir = ''
        self.vggwpath = 'vgg/vgg19pretrain.pdparams'
        self.output = 'output/'
        self.lastoutput = 'output/'
        self.predict_inst = 'prediction/test.png'
        self.predict_result = 'prediction/result.jpg'

        