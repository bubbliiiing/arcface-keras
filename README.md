## Arcface：人脸识别模型在Keras当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | accuracy | Validation rate |
| :-----: | :-----: | :------: | :------: | :------: | :------: |
| CASIA-WebFace | [arcface_mobilenet.h5](https://github.com/bubbliiiing/arcface-keras/releases/download/v1.0/arcface_mobilenet.h5) | LFW | 112x112 | 99.00% |  0.95200+-0.02237 @ FAR=0.00100 |
| CASIA-WebFace | [arcface_mobilefacenet.h5](https://github.com/bubbliiiing/arcface-keras/releases/download/v1.0/arcface_mobilefacenet.h5) | LFW | 112x112 | 99.02% | 0.96500+-0.01344 @ FAR=0.00133 |
| CASIA-WebFace | [arcface_iresnet50.h5](https://github.com/bubbliiiing/arcface-keras/releases/download/v1.0/arcface_iresnet50.h5) | LFW | 112x112 | 98.98% | 0.92967+-0.01935 @ FAR=0.00133 |

## 所需环境
tensorflow==1.13.2
keras=2.1.5

## 文件下载
已经训练好的权值可以在百度网盘下载。    
链接：https://pan.baidu.com/s/1P3-T6_PoXGTMYa_VuiwXmw 提取码: 114e

训练用的CASIA-WebFaces数据集以及评估用的LFW数据集可以在百度网盘下载。    
链接: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw 提取码: bcrq   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，可直接运行predict.py输入：
```python
img\1_001.jpg
img\1_002.jpg
```  
2. 也可以在百度网盘下载权值，放入model_data，修改arcface.py文件的model_path后，输入：
```python
img\1_001.jpg
img\1_002.jpg
```  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在arcface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，backbone对应主干特征提取网络**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
    #--------------------------------------------------------------------------#
    "model_path"        : "model_data/arcface_mobilefacenet.h5",
    #-------------------------------------------#
    #   输入图片的大小。
    #-------------------------------------------#
    "input_shape"       : [112, 112, 3],
    #-------------------------------------------#
    #   所使用到的主干特征提取网络，与训练的相同
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet50
    #-------------------------------------------#
    "backbone"          : "mobilefacenet",
    #-------------------------------------------#
    #   是否进行不失真的resize
    #-------------------------------------------#
    "letterbox_image"   : True,
}
```
3. 运行predict.py，输入  
```python
img\1_001.jpg
img\1_002.jpg
```  

## 训练步骤
1. 本文使用如下格式进行训练。
```
|-datasets
    |-people0
        |-123.jpg
        |-234.jpg
    |-people1
        |-345.jpg
        |-456.jpg
    |-...
```  
2. 下载好数据集，将训练用的CASIA-WebFaces数据集以及评估用的LFW数据集，解压后放在根目录。
3. 在训练前利用txt_annotation.py文件生成对应的cls_train.txt。  
4. 利用train.py训练模型，训练前，根据自己的需要选择backbone，model_path和backbone一定要对应。
5. 运行train.py即可开始训练。

## 评估步骤
1. 下载好评估数据集，将评估用的LFW数据集，解压后放在根目录
2. 在eval_LFW.py设置使用的主干特征提取网络和网络权值。
3. 运行eval_LFW.py来进行模型准确率评估。

## Reference
https://github.com/deepinsight/insightface  
https://github.com/timesler/facenet-pytorch   

