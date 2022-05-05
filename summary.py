#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.arcface import arcface
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape = [112, 112]
    backbone    = "mobilefacenet"
    model       = arcface([input_shape[0], input_shape[1], 3], 10575, backbone=backbone, mode="predict")
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
