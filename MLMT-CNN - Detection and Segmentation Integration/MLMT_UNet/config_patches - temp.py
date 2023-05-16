

train_dir_data = 'E:/Data/ARs/patches/train/'
val_dir_data = 'E:/Data/ARs/patches/validation/'



saving_dir = './model/patches/'

spect_1 = '284'
spect_2 = '195'
spect_3 = '171'
spect_4 = '304'


config = {

    'spect_1' : spect_1,
    'spect_2' : spect_2,
    'spect_3' : spect_3,
    'spect_4' : spect_4,

    'train_dir_img_1' : train_dir_data + '/' + spect_1 + '_patches_prime/images/',
    'train_dir_img_2' : train_dir_data + '/' + spect_2 + '_patches_prime/images/',
    'train_dir_img_3' : train_dir_data + '/' + spect_3 + '_patches_prime/images/',
    'train_dir_img_4' : train_dir_data + '/' + spect_4 + '_patches_prime/images/',

    'train_dir_seg_1' : train_dir_data + '/' + spect_1 + '_patches_prime/masks/',
    'train_dir_seg_2' : train_dir_data + '/' + spect_2 + '_patches_prime/masks/',
    'train_dir_seg_3' : train_dir_data + '/' + spect_3 + '_patches_prime/masks/',
    'train_dir_seg_4' : train_dir_data + '/' + spect_4 + '_patches_prime/masks/',



    'val_dir_img_1' : val_dir_data + '/' + spect_1 + '_patches_prime/images/',
    'val_dir_img_2' : val_dir_data + '/' + spect_2 + '_patches_prime/images/',
    'val_dir_img_3' : val_dir_data + '/' + spect_3 + '_patches_prime/images/',
    'val_dir_img_4' : val_dir_data + '/' + spect_4 + '_patches_prime/images/',

    'val_dir_seg_1' : val_dir_data + '/' + spect_1 + '_patches_prime/masks/',
    'val_dir_seg_2' : val_dir_data + '/' + spect_2 + '_patches_prime/masks/',
    'val_dir_seg_3' : val_dir_data + '/' + spect_3 + '_patches_prime/masks/',
    'val_dir_seg_4' : val_dir_data + '/' + spect_4 + '_patches_prime/masks/',

    'n_classes' : 3,

    'input_height'  : 224,
    'input_width'   : 224,
    'output_height' : 224,
    'output_width'  : 224,

    'GPU_fraction' : 0.70,

    'pre_trained_model' : 'weights.h5',

    'fine_tune' : '',

    'class_weights' : [2,1,2],

    'lr' : 0.004,

    'batch_size' : 8,

    'steps_per_epoch' : 240,

    'epochs' : 250,

    'saving_dir': saving_dir,

    'weights_path' : saving_dir + 'weights.{epoch:02d}_val_loss_{val_loss:.2f}.h5',

    'verbose' : 1,

    'model_summary' : False,

    'vis_at_test' : True

        }









