# Muti_Face_Recognition

### 运行环境
下面为软件的运行搭建系统环境

### 系统: window或linux
### 软件: python 3.x 、 tensorflow

### python支持库:

#### tensorflow:
pip install tensorflow      #cpu版本

pip install rensorflow-gpu  #gpu版本，需要cuda与cudnn的支持，不清楚的可以选择cpu版

#### numpy:
pip install numpy

#### opencv:
pip install opencv-python

#### dlib:
pip install dlib

### 设置人脸数据集：
set_face_database.py

### 训练神经网络：
train_CNN.py


### 人脸识别：
whose_face.py

此外，example中基于武林外传提供了一个已经训练好的预测器。能够识别出六位主角及其他人的七个分类，综合正确率约为90%。
![](https://github.com/yellowyellowyao/Muti_Face_Recognition/blob/master/example/rate.jpg)
