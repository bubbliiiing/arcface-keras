from keras import backend, initializers
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          DepthwiseConv2D, Dropout, Flatten,
                          GlobalAveragePooling2D, Multiply, PReLU, Reshape)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)

def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

def _bneck(inputs, expansion, out_ch, alpha, kernel_size, stride, se_ratio, activation, block_id, rate=1):
    in_channels     = backend.int_shape(inputs)[-1]
    exp_size        = _make_divisible(in_channels * expansion, 8)
    out_channels    = _make_divisible(out_ch * alpha, 8)
    
    x           = inputs
    prefix      = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(exp_size, 1, padding='same', use_bias=False, name=prefix + 'expand',
                      kernel_initializer=initializers.random_normal(stddev=0.1))(x)
        x = BatchNormalization(axis=-1, name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', dilation_rate=(rate, rate), use_bias=False, depthwise_initializer=initializers.random_normal(stddev=0.1), name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=-1, name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(x)
        y = Reshape([1, 1, exp_size], name=prefix + 'reshape')(y)
        
        y = Conv2D(reduced_ch, 1, padding='same', use_bias=True, name=prefix + 'squeeze_excite/Conv',
                      kernel_initializer=initializers.random_normal(stddev=0.1))(y)
        y = Activation("relu", name=prefix + 'squeeze_excite/Relu')(y)
        
        y = Conv2D(exp_size, 1, padding='same', use_bias=True, name=prefix + 'squeeze_excite/Conv_1',
                      kernel_initializer=initializers.random_normal(stddev=0.1))(y)
        x = Multiply(name=prefix + 'squeeze_excite/Mul')([Activation(hard_sigmoid)(y), x])

    x = Conv2D(out_channels, 1, padding='same', use_bias=False, name=prefix + 'project', 
                      kernel_initializer=initializers.random_normal(stddev=0.1))(x)
    x = BatchNormalization(axis=-1, name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = Add(name=prefix + 'Add')([inputs, x])
    return x

def MobilenetV3_small(inputs, embedding_size, dropout_keep_prob=0.5, alpha=1.0, kernel=5, se_ratio=0.25):
    x = Conv2D(16, 3, strides=(1, 1), padding='same', use_bias=False, name='Conv', 
                      kernel_initializer=initializers.random_normal(stddev=0.1))(inputs)
    x = BatchNormalization(axis=-1, name='Conv/BatchNorm')(x)
    x = Activation(hard_swish)(x)

    x = _bneck(x, 1, 16, alpha, 3, 2, se_ratio, 'relu', 0)
    
    x = _bneck(x, 4.5, 24, alpha, 3, 2, None, 'relu', 1)
    x = _bneck(x, 3.66, 24, alpha, 3, 1, None, 'relu', 2)
    
    x = _bneck(x, 4, 40, alpha, kernel, 2, se_ratio, 'hardswish', 3)
    x = _bneck(x, 6, 40, alpha, kernel, 1, se_ratio, 'hardswish', 4)
    x = _bneck(x, 6, 40, alpha, kernel, 1, se_ratio, 'hardswish', 5)
    x = _bneck(x, 3, 48, alpha, kernel, 1, se_ratio, 'hardswish', 6)
    x = _bneck(x, 3, 48, alpha, kernel, 1, se_ratio, 'hardswish', 7)
    
    x = _bneck(x, 6, 96, alpha, kernel, 2, se_ratio, 'hardswish', 8)
    x = _bneck(x, 6, 96, alpha, kernel, 1, se_ratio, 'hardswish', 9)
    x = _bneck(x, 6, 96, alpha, kernel, 1, se_ratio, 'hardswish', 10)

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

def MobileNetV3_Large(inputs, embedding_size, dropout_keep_prob=0.5, alpha=1.0, kernel=5, se_ratio=0.25):
    x = Conv2D(16, 3, strides=(1, 1), padding='same', use_bias=False, name='Conv', 
                      kernel_initializer=initializers.random_normal(stddev=0.1))(inputs)
    x = BatchNormalization(axis=-1, name='Conv/BatchNorm')(x)
    x = Activation(hard_swish)(x)

    x = _bneck(x, 1, 16, alpha, 3, 1, None, 'relu', 0)

    x = _bneck(x, 4, 24, alpha, 3, 2, None, 'relu', 1)
    x = _bneck(x, 3, 24, alpha, 3, 1, None, 'relu', 2)
    
    x = _bneck(x, 3, 40, alpha, kernel, 2, se_ratio, 'relu', 3)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 4)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 5)
    
    x = _bneck(x, 6, 80, alpha, 3, 2, None, 'hardswish', 6)
    x = _bneck(x, 2.5, 80, alpha, 3, 1, None, 'hardswish', 7)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 8)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 9)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 10)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 11)
    
    x = _bneck(x, 6, 160, alpha, kernel, 2, se_ratio, 'hardswish', 12)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 13)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 14)
    
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
