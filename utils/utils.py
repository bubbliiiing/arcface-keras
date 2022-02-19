import math

import numpy as np
import tensorflow as tf
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

def preprocess_input(image):
    image /= 255.0 
    image -= 0.5
    image /= 0.5
    return image

def get_acc(s=32.0, m=0.5):
    cos_m      = math.cos(m)
    sin_m      = math.sin(m)
    th         = math.cos(math.pi - m)
    mm         = math.sin(math.pi - m) * m
    def acc(y_true, y_pred):
        cosine = tf.cast(y_pred, tf.float32)
        labels = tf.cast(y_true, tf.float32)

        sine = tf.sqrt(1 - tf.square(cosine))
        phi = cosine * cos_m - sine * sin_m
        phi = tf.where(cosine > th, phi, cosine - mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= s

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.math.softmax(output, -1), -1), tf.argmax(y_true, -1)), tf.float32))
        return accuracy
    return acc
