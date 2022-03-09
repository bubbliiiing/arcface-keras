#-------------------------------------------------------------#
#   MobileNetV2的网络部分
#-------------------------------------------------------------#

from keras import backend
from keras import initializers
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          DepthwiseConv2D, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, PReLU, Reshape,
                          ZeroPadding2D)
from keras.layers.normalization import BatchNormalization


def relu6(x):
    return backend.relu(x, max_value=6)
    
def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    x = inputs
    prefix = 'block_{}_'.format(block_id)
    #---------------------------------------------#
    #   part1 数据扩张
    #---------------------------------------------#
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=initializers.random_normal(stddev=0.1),
                          activation=None,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                                 name=prefix + 'pad')(x)
    
    #---------------------------------------------#
    #   part2 可分离卷积
    #---------------------------------------------#
    x = DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               depthwise_initializer=initializers.random_normal(stddev=0.1),
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    #---------------------------------------------#
    #   part3压缩特征，而且不使用relu函数，保证特征不被破坏
    #---------------------------------------------#
    x = Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=initializers.random_normal(stddev=0.1),
                      activation=None,
                      name=prefix + 'project')(x)

    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def MobilenetV2(inputs, embedding_size, dropout_keep_prob=0.5, alpha=1.0):
    #---------------------------------------------#
    #   stem部分
    #---------------------------------------------#
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3),
                             name='Conv1_pad')(inputs)
    
    x = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(1, 1),
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=initializers.random_normal(stddev=0.1),
                      name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)


    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)


    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)


    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)
    
    x = Conv2D(512, kernel_size=1, use_bias=False, name='sep',
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name='sep_bn', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)

    x = BatchNormalization(name='bn2', epsilon=1e-5)(x)
    x = Dropout(p=dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, name='linear',
            kernel_initializer=initializers.random_normal(stddev=0.1),
            bias_initializer='zeros')(x)
    x = BatchNormalization(name='features', epsilon=1e-5)(x)
    return x
