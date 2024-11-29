

train_dir_data = 'E:/Data/ARs/full_disk/train/'
val_dir_data = 'E:/Data/ARs/full_disk/validation/'

saving_dir = './model/full_disk/'


spect_1 = '284'
spect_2 = '195'
spect_3 = '171'
spect_4 = '304'



config = {

    'spect_1' : spect_1,
    'spect_2' : spect_2,
    'spect_3' : spect_3,
    'spect_4' : spect_4,

    'train_dir_img_1' : train_dir_data + '/' + spect_1 + '/images/',
    'train_dir_img_2' : train_dir_data + '/' + spect_2 + '/images/',
    'train_dir_img_3' : train_dir_data + '/' + spect_3 + '/images/',
    'train_dir_img_4' : train_dir_data + '/' + spect_4 + '/images/',

    'train_dir_seg_1' : train_dir_data + '/' + spect_1 + '/masks_recursive_1/',
    'train_dir_seg_2' : train_dir_data + '/' + spect_2 + '/masks_recursive_1/',
    'train_dir_seg_3' : train_dir_data + '/' + spect_3 + '/masks_recursive_1/',
    'train_dir_seg_4' : train_dir_data + '/' + spect_4 + '/masks_recursive_1/',



    'val_dir_img_1' : val_dir_data + '/' + spect_1 + '/images/',
    'val_dir_img_2' : val_dir_data + '/' + spect_2 + '/images/',
    'val_dir_img_3' : val_dir_data + '/' + spect_3 + '/images/',
    'val_dir_img_4' : val_dir_data + '/' + spect_4 + '/images/',

    'val_dir_seg_1' : val_dir_data + '/' + spect_1 + '/masks/',
    'val_dir_seg_2' : val_dir_data + '/' + spect_2 + '/masks/',
    'val_dir_seg_3' : val_dir_data + '/' + spect_3 + '/masks/',
    'val_dir_seg_4' : val_dir_data + '/' + spect_4 + '/masks/',

    'n_classes' : 3,

    'input_height'  : 512,
    'input_width'   : 512,
    'output_height' : 512,
    'output_width'  : 512,

    'GPU_fraction' : 1.0,

    'pre_trained_model' : 'weights.h5',

    'fine_tune' : '',

    'class_weights' : [2,1,5],

    'lr' : 0.004,

    'batch_size' : 2,

    'steps_per_epoch' : 240,

    'epochs' : 250,

    'saving_dir': saving_dir,

    'weights_path' : saving_dir + 'weights.{epoch:02d}_val_loss_{val_loss:.2f}.h5',

    'verbose' : 1,

    'model_summary' : False,

    'vis_at_test' : True

        }









