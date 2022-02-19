from keras import initializers, layers
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          PReLU, ZeroPadding2D)
from keras.models import Model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2', epsilon=1e-5)(input_tensor)

    #----------------------------#
    #   减少通道数
    #----------------------------#
    x = Conv2D(filters1, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2a',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2a', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)

    #----------------------------#
    #   3x3卷积
    #----------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2b',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2b', epsilon=1e-5)(x)
    
    x = layers.add([x, input_tensor])
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2', epsilon=1e-5)(input_tensor)
    
    #----------------------------#
    #   减少通道数
    #----------------------------#
    x = Conv2D(filters1, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2a',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2a', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)

    #----------------------------#
    #   3x3卷积
    #----------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, strides=strides, name=conv_name_base + '2b',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2b', epsilon=1e-5)(x)
    
    #----------------------------#
    #   残差边
    #----------------------------#
    shortcut = Conv2D(filters2, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '1',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1', epsilon=1e-5)(shortcut)

    x = layers.add([x, shortcut])
    return x

def iResNet50(inputs, embedding_size, dropout_keep_prob=0.5):
    x = ZeroPadding2D((1, 1))(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), name='conv1', use_bias=False,
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name='bn_conv1', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)

    x = conv_block(x, 3, [64, 64], stage=2, block='a')
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
    x = identity_block(x, 3, [64, 64], stage=2, block='c')

    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
    x = identity_block(x, 3, [128, 128], stage=3, block='c')
    x = identity_block(x, 3, [128, 128], stage=3, block='d')

    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
    x = identity_block(x, 3, [256, 256], stage=4, block='c')
    x = identity_block(x, 3, [256, 256], stage=4, block='d')
    x = identity_block(x, 3, [256, 256], stage=4, block='e')
    x = identity_block(x, 3, [256, 256], stage=4, block='f')

    x = identity_block(x, 3, [256, 256], stage=4, block='g')
    x = identity_block(x, 3, [256, 256], stage=4, block='h')
    x = identity_block(x, 3, [256, 256], stage=4, block='i')
    x = identity_block(x, 3, [256, 256], stage=4, block='j')
    x = identity_block(x, 3, [256, 256], stage=4, block='k')

    x = identity_block(x, 3, [256, 256], stage=4, block='l')
    x = identity_block(x, 3, [256, 256], stage=4, block='m')
    x = identity_block(x, 3, [256, 256], stage=4, block='n')

    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
    x = identity_block(x, 3, [512, 512], stage=5, block='c')
    
    x = BatchNormalization(name='bn2', epsilon=1e-5)(x)
    x = Dropout(p=dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, name='linear',
            kernel_initializer=initializers.random_normal(stddev=0.1),
            bias_initializer='zeros')(x)
    x = BatchNormalization(name='features', epsilon=1e-5,)(x)

    return x
