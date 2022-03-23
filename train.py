import datetime
import os

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, DepthwiseConv2D, PReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

from nets.arcface import arcface
from nets.arcface_training import ArcFaceLoss, get_lr_scheduler
from utils.callbacks import LFW_callback, LossHistory
from utils.dataloader import FacenetDataset, LFWDataset
from utils.utils import get_acc, get_num_classes

if __name__ == "__main__":
    #--------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    #--------------------------------------------------------#
    annotation_path = "cls_train.txt"
    #--------------------------------------------------------#
    #   输入图像大小
    #--------------------------------------------------------#
    input_shape     = [112, 112, 3]
    #--------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   mobilenetv2
    #   mobilenetv3
    #   iresnet50
    #
    #   除了mobilenetv1外，其它的backbone均可从0开始训练。
    #--------------------------------------------------------#
    backbone        = "mobilefacenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = 主干的权值。
    #   如果想要让模型从0开始训练，则设置model_path = ''，此时从0开始训练。
    #----------------------------------------------------------------------------------------------------------------------------#  
    model_path      = ""

    #------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #------------------------------------------------------#
    #------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   batch_size      每次输入的图片数量
    #   Epoch           模型总共训练的epoch
    #------------------------------------------------------#
    batch_size      = 64
    Init_Epoch      = 0
    Epoch           = 50

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者1 
    #------------------------------------------------------------------#
    num_workers     = 1
    #------------------------------------------------------------------#
    #   是否开启LFW评估
    #------------------------------------------------------------------#
    lfw_eval_flag   = True
    #------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    #------------------------------------------------------------------#
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"

    num_classes = get_num_classes(annotation_path)
    #-------------------------------------------#
    #   建立模型
    #-------------------------------------------#
    model = arcface(input_shape, num_classes, backbone=backbone, mode="train")
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(weight_decay)(layer.kernel))
        elif isinstance(layer, PReLU):
            layer.add_loss(l2(weight_decay)(layer.alpha))

    if True:
        #-------------------------------------------------------------------#
        #   判断当前batch_size与64的差别，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs     = 64
        Init_lr = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr  = max(batch_size / nbs * Min_lr, 1e-6)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        m = 0.5
        s = 32 if backbone == "mobilefacenet" else 64
        model.compile(optimizer = optimizer, loss={'ArcMargin': ArcFaceLoss(s = s, m = m)}, metrics={'ArcMargin': get_acc()})
    
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataset   = FacenetDataset(input_shape, lines[:num_train], batch_size, num_classes, random = True)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], batch_size, num_classes, random = False)

        #-------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler       用于设置学习率下降的方式
        #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
        #-------------------------------------------------------------------------------#
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        #---------------------------------#
        #   LFW估计
        #---------------------------------#
        if lfw_eval_flag:
            lfw_callback    = LFW_callback(LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, batch_size=32, input_shape=input_shape))
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler, lfw_callback]
        else:
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataset,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataset,
            validation_steps    = epoch_step_val,
            epochs              = Epoch,
            initial_epoch       = Init_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = callbacks
        )
