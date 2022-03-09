#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.arcface import arcface

if __name__ == "__main__":
    input_shape = [112, 112, 3]
    model       = arcface(input_shape, 10575, backbone="mobilenetv2", mode="predict")
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
