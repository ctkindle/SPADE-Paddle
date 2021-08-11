# set up global parameters
class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        self.D_steps_per_G=1
        self.aspect_ratio=1.0
        self.batchSize=1
        self.beta1=0.0
        self.beta2=0.999
        self.cache_filelist_read=True
        self.cache_filelist_write=True
        self.change_min=0.1
        self.checkpoints_dir='./checkpoints'
        self.coco_no_portraits=False
        self.contain_dontcare_label=True
        self.continue_train=False
        self.crop_size=256
        self.dataset_mode='coco'
        self.debug=False
        self.display_freq=100
        self.display_winsize=256
        self.gan_mode='hinge'
        self.gpu_ids=[]
        self.init_type='xavier'
        self.init_variance=0.02
        self.isTrain=True
        self.label_nc=182
        self.lambda_feat=10.0
        self.lambda_kld=0.05
        self.lambda_mask=100.0
        self.lambda_vgg=0.4
        self.load_from_opt_file=False
        self.load_size=286
        self.lr=0.0001
        self.max_dataset_size=9223372036854775807
        self.model='pix2pix'
        self.nThreads=0
        self.n_layers_D=4
        self.name='label2coco'
        self.ndf=64
        self.nef=16
        self.netD='multiscale'
        self.netD_subarch='n_layer'
        self.netG='spade'
        self.ngf=64
        self.niter=50
        self.niter_decay=0
        self.no_TTUR=False
        self.no_flip=False
        self.no_ganFeat_loss=False
        self.no_html=False
        self.no_instance=False
        self.no_pairing_check=False
#         self.no_vgg_loss=False
        self.no_vgg_loss=True
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
#         self.norm_G='spectralspadesyncbatch3x3'
        self.norm_G='spectralspadebatch3x3'
        self.num_D=2
        self.num_upsampling_layers='normal'
        self.optimizer='adam'
        self.output_nc=3
        self.phase='train'
        self.preprocess_mode='resize_and_crop'
        self.print_freq=100
        self.save_epoch_freq=10
        self.save_latest_freq=5000
        self.semantic_nc=184
        self.serial_batches=False
        self.tf_log=False
        self.use_vae=True
        self.which_epoch='latest'
        self.z_dim=256
        self.dataroot = 'dataset/'
        self.datasetdir = ''
        self.vggwpath = 'vgg/vgg19pretrain.pdparams'
        self.output = 'output/'
        self.lastoutput = 'output/'

        