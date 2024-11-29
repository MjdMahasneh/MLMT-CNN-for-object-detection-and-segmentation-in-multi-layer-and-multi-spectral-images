from keras.models import *
from keras.layers import *



def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)

    if batchnorm:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)

    if batchnorm:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x



def get_mlmt_unet(n_classes, n_filters = 16, dropout = 0.05, batchnorm = True, input_height = None, input_width = None, merge=None):
    assert merge == 'add' or merge == 'cat', ('expecting add or cat!')
    assert input_width != None and input_width != None
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input_spect_1 = Input(shape=(None, None, 3))
    img_input_spect_2 = Input(shape=(None, None, 3))
    img_input_spect_3 = Input(shape=(None, None, 3))
    img_input_spect_4 = Input(shape=(None, None, 3))

    # contracting path 1
    c1_spect1 = conv2d_block(img_input_spect_1, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1_spect1 = MaxPooling2D((2, 2))(c1_spect1)
    p1_spect1 = Dropout(dropout * 0.5)(p1_spect1)

    c2_spect1 = conv2d_block(p1_spect1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2_spect1 = MaxPooling2D((2, 2))(c2_spect1)
    p2_spect1 = Dropout(dropout)(p2_spect1)

    c3_spect1 = conv2d_block(p2_spect1, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3_spect1 = MaxPooling2D((2, 2))(c3_spect1)
    p3_spect1 = Dropout(dropout)(p3_spect1)

    c4_spect1 = conv2d_block(p3_spect1, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4_spect1 = MaxPooling2D(pool_size=(2, 2))(c4_spect1)
    p4_spect1 = Dropout(dropout)(p4_spect1)

    c5_spect1 = conv2d_block(p4_spect1, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # contracting path 2
    c1_spect2 = conv2d_block(img_input_spect_2, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1_spect2 = MaxPooling2D((2, 2))(c1_spect2)
    p1_spect2 = Dropout(dropout * 0.5)(p1_spect2)

    c2_spect2 = conv2d_block(p1_spect2, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2_spect2 = MaxPooling2D((2, 2))(c2_spect2)
    p2_spect2 = Dropout(dropout)(p2_spect2)

    c3_spect2 = conv2d_block(p2_spect2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3_spect2 = MaxPooling2D((2, 2))(c3_spect2)
    p3_spect2 = Dropout(dropout)(p3_spect2)

    c4_spect2 = conv2d_block(p3_spect2, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4_spect2 = MaxPooling2D(pool_size=(2, 2))(c4_spect2)
    p4_spect2 = Dropout(dropout)(p4_spect2)

    c5_spect2 = conv2d_block(p4_spect2, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # contracting path 3
    c1_spect3 = conv2d_block(img_input_spect_3, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1_spect3 = MaxPooling2D((2, 2))(c1_spect3)
    p1_spect3 = Dropout(dropout * 0.5)(p1_spect3)

    c2_spect3 = conv2d_block(p1_spect3, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2_spect3 = MaxPooling2D((2, 2))(c2_spect3)
    p2_spect3 = Dropout(dropout)(p2_spect3)

    c3_spect3 = conv2d_block(p2_spect3, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3_spect3 = MaxPooling2D((2, 2))(c3_spect3)
    p3_spect3 = Dropout(dropout)(p3_spect3)

    c4_spect3 = conv2d_block(p3_spect3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4_spect3 = MaxPooling2D(pool_size=(2, 2))(c4_spect3)
    p4_spect3 = Dropout(dropout)(p4_spect3)

    c5_spect3 = conv2d_block(p4_spect3, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # contracting path 4
    c1_spect4 = conv2d_block(img_input_spect_4, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1_spect4 = MaxPooling2D((2, 2))(c1_spect4)
    p1_spect4 = Dropout(dropout * 0.5)(p1_spect4)

    c2_spect4 = conv2d_block(p1_spect4, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2_spect4 = MaxPooling2D((2, 2))(c2_spect4)
    p2_spect4 = Dropout(dropout)(p2_spect4)

    c3_spect4 = conv2d_block(p2_spect4, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3_spect4 = MaxPooling2D((2, 2))(c3_spect4)
    p3_spect4 = Dropout(dropout)(p3_spect4)

    c4_spect4 = conv2d_block(p3_spect4, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4_spect4 = MaxPooling2D(pool_size=(2, 2))(c4_spect4)
    p4_spect4 = Dropout(dropout)(p4_spect4)

    c5_spect4 = conv2d_block(p4_spect4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # merge
    if merge == 'add':
        merged = Add()([c5_spect1, c5_spect2, c5_spect3, c5_spect4])
    elif merge == 'cat':
        merged = concatenate([c5_spect1, c5_spect2, c5_spect3, c5_spect4])


    # expansive path 1
    u6_spect1 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(merged)
    u6_spect1 = concatenate([u6_spect1, c4_spect1])
    u6_spect1 = Dropout(dropout)(u6_spect1)
    c6_spect1 = conv2d_block(u6_spect1, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7_spect1 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6_spect1)
    u7_spect1 = concatenate([u7_spect1, c3_spect1])
    u7_spect1 = Dropout(dropout)(u7_spect1)
    c7_spect1 = conv2d_block(u7_spect1, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8_spect1 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7_spect1)
    u8_spect1 = concatenate([u8_spect1, c2_spect1])
    u8_spect1 = Dropout(dropout)(u8_spect1)
    c8_spect1 = conv2d_block(u8_spect1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9_spect1 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8_spect1)
    u9_spect1 = concatenate([u9_spect1, c1_spect1], axis=3)
    u9_spect1 = Dropout(dropout)(u9_spect1)
    c9_spect1 = conv2d_block(u9_spect1, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # expansive path 2
    u6_spect2 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(merged)
    u6_spect2 = concatenate([u6_spect2, c4_spect2])
    u6_spect2 = Dropout(dropout)(u6_spect2)
    c6_spect2 = conv2d_block(u6_spect2, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7_spect2 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6_spect2)
    u7_spect2 = concatenate([u7_spect2, c3_spect2])
    u7_spect2 = Dropout(dropout)(u7_spect2)
    c7_spect2 = conv2d_block(u7_spect2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8_spect2 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7_spect2)
    u8_spect2 = concatenate([u8_spect2, c2_spect2])
    u8_spect2 = Dropout(dropout)(u8_spect2)
    c8_spect2 = conv2d_block(u8_spect2, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9_spect2 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8_spect2)
    u9_spect2 = concatenate([u9_spect2, c1_spect2], axis=3)
    u9_spect2 = Dropout(dropout)(u9_spect2)
    c9_spect2 = conv2d_block(u9_spect2, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # expansive path 3
    u6_spect3 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(merged)
    u6_spect3 = concatenate([u6_spect3, c4_spect3])
    u6_spect3 = Dropout(dropout)(u6_spect3)
    c6_spect3 = conv2d_block(u6_spect3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7_spect3 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6_spect3)
    u7_spect3 = concatenate([u7_spect3, c3_spect3])
    u7_spect3 = Dropout(dropout)(u7_spect3)
    c7_spect3 = conv2d_block(u7_spect3, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8_spect3 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7_spect3)
    u8_spect3 = concatenate([u8_spect3, c2_spect3])
    u8_spect3 = Dropout(dropout)(u8_spect3)
    c8_spect3 = conv2d_block(u8_spect3, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9_spect3 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8_spect3)
    u9_spect3 = concatenate([u9_spect3, c1_spect3], axis=3)
    u9_spect3 = Dropout(dropout)(u9_spect3)
    c9_spect3 = conv2d_block(u9_spect3, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # expansive path 4
    u6_spect4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(merged)
    u6_spect4 = concatenate([u6_spect4, c4_spect4])
    u6_spect4 = Dropout(dropout)(u6_spect4)
    c6_spect4 = conv2d_block(u6_spect4, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7_spect4 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6_spect4)
    u7_spect4 = concatenate([u7_spect4, c3_spect4])
    u7_spect4 = Dropout(dropout)(u7_spect4)
    c7_spect4 = conv2d_block(u7_spect4, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8_spect4 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7_spect4)
    u8_spect4 = concatenate([u8_spect4, c2_spect4])
    u8_spect4 = Dropout(dropout)(u8_spect4)
    c8_spect4 = conv2d_block(u8_spect4, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9_spect4 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8_spect4)
    u9_spect4 = concatenate([u9_spect4, c1_spect4], axis=3)
    u9_spect4 = Dropout(dropout)(u9_spect4)
    c9_spect4 = conv2d_block(u9_spect4, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)


    # output heads :
    output_spect_1 = Conv2D(n_classes, (1, 1), activation='softmax')(c9_spect1)

    output_spect_2 = Conv2D(n_classes, (1, 1), activation='softmax')(c9_spect2)

    output_spect_3 = Conv2D(n_classes, (1, 1), activation='softmax')(c9_spect3)

    output_spect_4 = Conv2D(n_classes, (1, 1), activation='softmax')(c9_spect4)


    model = Model(inputs=[img_input_spect_1, img_input_spect_2, img_input_spect_3, img_input_spect_4],
                  outputs=[output_spect_1, output_spect_2, output_spect_3, output_spect_4])

    return model