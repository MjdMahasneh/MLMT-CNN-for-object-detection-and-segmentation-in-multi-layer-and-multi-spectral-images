from __future__ import print_function
from __future__ import absolute_import
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed
from MLMT_RCNN.roi_pooling_conv import RoiPoolingConv
from MLMT_RCNN.fixed_batch_normalization import FixedBatchNormalization
from keras import backend as K
import keras
from keras.models import Model

def get_img_output_length(width, height):
    def get_output_length(input_length):
        input_length += 6
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length
    return get_output_length(width), get_output_length(height)

def load_nested_imagenet(model):
    pre_trained_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
    for layer in model.layers:
        if 'resnet50' in layer.name:
            for nested_layer in layer.layers:
                for layer_pre_trained in pre_trained_model.layers:
                    if layer_pre_trained.name in nested_layer.name:
                        try:
                            nested_layer.set_weights(layer_pre_trained.get_weights())
                        except Exception as e:
                            pass
        else:
            for layer_pre_trained in pre_trained_model.layers:
                if layer_pre_trained.name in layer.name:
                    try:
                        layer.set_weights(layer_pre_trained.get_weights())
                    except Exception as e:
                        pass
    return model




def identity_block(input_tensor, kernel_size, filters, stage, block, ResBranch, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a' + '_resBranch_' + ResBranch, trainable=trainable)(
        input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b' + '_resBranch_' + ResBranch,
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c' + '_resBranch_' + ResBranch, trainable=trainable)(
        x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c' + '_resBranch_' + ResBranch)(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, ResBranch, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a' + '_resBranch_' + ResBranch)(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                      padding='same'), name=conv_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c' + '_resBranch_' + ResBranch)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c' + '_resBranch_' + ResBranch)(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, ResBranch, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a' + '_resBranch_' + ResBranch,
                      trainable=trainable)(
        input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b' + '_resBranch_' + ResBranch,
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c' + '_resBranch_' + ResBranch, trainable=trainable)(
        x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c' + '_resBranch_' + ResBranch)(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1' + '_resBranch_' + ResBranch,
                             trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1' + '_resBranch_' + ResBranch)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, ResBranch, input_shape, strides=(2, 2),
                  trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(
        Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        input_shape=input_shape, name=conv_name_base + '2a' + '_resBranch_' + ResBranch)(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                                      kernel_initializer='normal'),
                        name=conv_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b' + '_resBranch_' + ResBranch)(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'),
                        name=conv_name_base + '2c' + '_resBranch_' + ResBranch,
                        trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c' + '_resBranch_' + ResBranch)(x)

    shortcut = TimeDistributed(
        Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1' + '_resBranch_' + ResBranch)(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis),
                               name=bn_name_base + '1' + '_resBranch_' + ResBranch)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def mlmt_base_nn(input_tensor_1 = None, input_tensor_2 = None, trainable=False):

    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor_1 is None:
        img_input_1 = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor_1):
            img_input_1 = Input(tensor=input_tensor_1, shape=input_shape)
        else:
            img_input_1 = input_tensor_1

    if input_tensor_2 is None:
        img_input_2 = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor_2):
            img_input_2 = Input(tensor=input_tensor_2, shape=input_shape)
        else:
            img_input_2 = input_tensor_2

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    ## Branch 1
    x_1 = ZeroPadding2D((3, 3))(img_input_1)

    x_1 = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1_brch_1', trainable=trainable)(x_1)

    x_1 = FixedBatchNormalization(axis=bn_axis, name='bn_conv1_brch_1')(x_1)
    x_1 = Activation('relu')(x_1)
    x_1 = MaxPooling2D((3, 3), strides=(2, 2))(x_1)

    x_1 = conv_block(x_1, 3, [64, 64, 256], stage=2, block='a', ResBranch = 'brch_1', strides=(1, 1), trainable=trainable)
    x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='b', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='c', ResBranch = 'brch_1', trainable=trainable)

    x_1 = conv_block(x_1, 3, [128, 128, 512], stage=3, block='a', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='b', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='c', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='d', ResBranch = 'brch_1', trainable=trainable)

    x_1 = conv_block(x_1, 3, [256, 256, 1024], stage=4, block='a', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='b', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='c', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='d', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='e', ResBranch = 'brch_1', trainable=trainable)
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='f', ResBranch = 'brch_1', trainable=trainable)

    ## Branch 2
    x_2 = ZeroPadding2D((3, 3))(img_input_2)

    x_2 = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1_brch_2', trainable=trainable)(x_2)

    x_2 = FixedBatchNormalization(axis=bn_axis, name='bn_conv1_brch_2')(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = MaxPooling2D((3, 3), strides=(2, 2))(x_2)

    x_2 = conv_block(x_2, 3, [64, 64, 256], stage=2, block='a', ResBranch = 'brch_2', strides=(1, 1), trainable=trainable)
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='b', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='c', ResBranch = 'brch_2', trainable=trainable)

    x_2 = conv_block(x_2, 3, [128, 128, 512], stage=3, block='a', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='b', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='c', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='d', ResBranch = 'brch_2', trainable=trainable)

    x_2 = conv_block(x_2, 3, [256, 256, 1024], stage=4, block='a', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='b', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='c', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='d', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='e', ResBranch = 'brch_2', trainable=trainable)
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='f', ResBranch = 'brch_2', trainable=trainable)

    resnet_1 = Model(img_input_1, x_1, name='resnet50_1')
    resnet_2 = Model(img_input_2, x_2, name='resnet50_2')

    x_1 = resnet_1(img_input_1)
    x_2 = resnet_2(img_input_2)

    merge = keras.layers.concatenate([x_1, x_2])
    return merge, x_1, x_2

def classifier_layers(x, spect, input_shape, trainable=False):
    ResBrn = 'cls_lyr'+'_spect'+spect
    if K.backend() == 'tensorflow':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', ResBranch = ResBrn , input_shape=input_shape, strides=(2, 2),
                          trainable=trainable)
    elif K.backend() == 'theano':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', ResBranch = ResBrn, input_shape=input_shape, strides=(1, 1),
                          trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', ResBranch = ResBrn, trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', ResBranch = ResBrn, trainable=trainable)
    avg_pool = 'avg_pool'+'_'+spect
    x = TimeDistributed(AveragePooling2D((7, 7)), name = avg_pool)(x)
    return x

def rpn(base_layers_merged, base_layers_x1, base_layers_x2, num_anchors):
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv2')(base_layers_merged)

    x_1_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class_1')(x)
    x_1_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress_1')(x)

    x_2_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class_2')(x)
    x_2_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress_2')(x)
    return [x_1_class, x_1_regr, x_2_class, x_2_regr, base_layers_x1, base_layers_x2]

def roiconvmulspect(x1, x2, rois_1, rois_2, num_rois, pooling_regions):
    out_roi_pool_x1_r1 = RoiPoolingConv(pooling_regions, num_rois)([x1, rois_1])
    out_roi_pool_x2_r1 = RoiPoolingConv(pooling_regions, num_rois)([x2, rois_1])
    merged_rois_r1 = keras.layers.concatenate([out_roi_pool_x1_r1, out_roi_pool_x2_r1])
    out_roi_pool_x1_r2 = RoiPoolingConv(pooling_regions, num_rois)([x1, rois_2])
    out_roi_pool_x2_r2 = RoiPoolingConv(pooling_regions, num_rois)([x2, rois_2])
    merged_rois_r2 = keras.layers.concatenate([out_roi_pool_x1_r2, out_roi_pool_x2_r2])
    return merged_rois_r1, merged_rois_r2

def classifier(base_layers_x1, base_layers_x2, input_rois_1, input_rois_2, num_rois, nb_classes=2, trainable=False):

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 2048)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 2048, 7, 7)

    out_roi_pool_1, out_roi_pool_2 = roiconvmulspect(base_layers_x1, base_layers_x2, input_rois_1, input_rois_2, num_rois, pooling_regions)

    out_1 = classifier_layers(out_roi_pool_1, spect = '1', input_shape=input_shape, trainable=trainable)
    out_2 = classifier_layers(out_roi_pool_2, spect = '2', input_shape=input_shape, trainable=trainable)

    out_1 = TimeDistributed(Flatten())(out_1)
    out_2 = TimeDistributed(Flatten())(out_2)

    out_class_1 = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_1{}'.format(nb_classes))(out_1)
    out_regr_1 = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_1{}'.format(nb_classes))(out_1)

    out_class_2 = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_2{}'.format(nb_classes))(out_2)
    out_regr_2 = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_2{}'.format(nb_classes))(out_2)

    return [out_class_1, out_regr_1, out_class_2, out_regr_2]

