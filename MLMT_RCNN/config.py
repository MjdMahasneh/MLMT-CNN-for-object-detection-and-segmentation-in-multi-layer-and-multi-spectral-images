from keras import backend as K


class Config:
    def __init__(self):

        self.im_size = 600

        self.use_horizontal_flips = False
        
        self.use_vertical_flips = False
        
        self.rot_90 = False

        self.saving_path = './model/'

        self.spect_1 = '195'
        
        self.spect_2 = '171'

        self.training_images_dir = 'F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/'

        self.simple_label_file_A = 'label_a.txt'
        
        self.simple_label_file_B = 'label_b.txt'

        self.config_save_file = 'config.pickle'

        self.gpu_fraction = 0.65

        self.num_epochs = 6000

        self.verbose = True

        self.num_rois = 64

        self.anchor_box_scales = [32, 64, 128, 256]

        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        self.finetune = False

        self.img_channel_mean = [103.939, 116.779, 123.68]
        
        self.img_scaling_factor = 1.0

        self.rpn_stride = 16

        self.balanced_classes = False

        self.std_scaling = 4.0
        
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.rpn_min_overlap = 0.3
        
        self.rpn_max_overlap = 0.7

        self.classifier_min_overlap = 0.1
        
        self.classifier_max_overlap = 0.5

        self.class_mapping = None





