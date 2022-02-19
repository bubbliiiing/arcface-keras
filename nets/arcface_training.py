import math
from functools import partial

import keras.backend as K
import tensorflow as tf


class ArcFaceLoss() :
    def __init__(self, s=32.0, m=0.5) :
        self.s = s

        self.cos_m      = math.cos(m)
        self.sin_m      = math.sin(m)
        
        self.th         = math.cos(math.pi - m)
        self.mm         = math.sin(math.pi - m) * m

    def __call__(self, y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        cosine = tf.cast(y_pred, tf.float32)
        #----------------------------------------------------#
        #   batch_size, 10575 -> batch_size, 10575
        #----------------------------------------------------#
        sine    = tf.sqrt(1 - tf.square(cosine))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = tf.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        
        losses = K.categorical_crossentropy(y_true, output, from_logits=True)
        # losses = tf.Print(losses,[tf.shape(losses),tf.shape(y_true),tf.shape(output)])
        return losses

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

