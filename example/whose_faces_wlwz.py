# coding=utf-8
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys

# 载入预测器模型，默认选择最新的预测器
model_path = './train_faces.model-18299'
# model_path = tf.train.latest_checkpoint('.')

# 预测时网络要求与训练时网络结构一致!!!
# 若载入之前的预测器，请保持网络结构一致。

test_faces1 = r"./faces"


test_path = [test_faces1]

faces = ["Unknown",
         "Txy",
         "Bzt",
         "Gfr",
         "Ldz",
         "Mxb",
         "Lxc"]


size = 64

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, len(faces)])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    # stddev 正态分布标准差，默认1.0
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, W):
    # 计算给定4-D input和filter张量的2-D卷积。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnLayer():
    # 第一层
    W1 = weightVariable([3, 3, 3, 32])
    # 卷积核大小(3,3)， 输入通道数(3)， 输出通道数(32)
    b1 = biasVariable([32])
    # 卷积  计算纠正线性：max(features, 0)。
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化 对输入执行最大池化
    pool1 = maxPool(conv1)
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    # pool2 =tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    # W3 = weightVariable([3, 3, 64, 64])
    # b3 = biasVariable([64])
    W3 = weightVariable([3, 3, 64, 128])
    b3 = biasVariable([128])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    # pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    drop3 = dropout(pool3, keep_prob_5)

    # 第四层
    # W4 = weightVariable([3, 3, 64, 64])
    # b4 = biasVariable([64])
    W4 = weightVariable([3,3,128,256])
    b4 = biasVariable([256])
    conv4 = tf.nn.relu(conv2d(drop3, W4) + b4)
    # pool4 = maxPool(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    drop4 = dropout(pool4, keep_prob_5)

    # 第五层
    # W5 = weightVariable([3, 3, 64, 64])
    # b5 = biasVariable([64])
    W5 = weightVariable([3,3,256,256])
    b5 = biasVariable([256])
    conv5 = tf.nn.relu(conv2d(drop4, W5) + b5)
    # pool5 = maxPool(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    drop5 = dropout(pool5, keep_prob_5)

    # # 第六层
    # W6 = weightVariable([3,3,64,64])
    # b6 = biasVariable([64])
    # conv6 = tf.nn.relu(conv2d(drop5, W6) + b6)
    # # pool6 = maxPool(conv6)
    # pool6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop6 = dropout(pool6, keep_prob_5)
    #
    # # 第七层
    # W7 = weightVariable([3,3,64,64])
    # b7 = biasVariable([64])
    # conv7 = tf.nn.relu(conv2d(drop6, W7) + b7)
    # # pool7 = maxPool(conv7)
    # pool7 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop7 = dropout(pool7, keep_prob_5)
    #
    # # 第8层
    # W8 = weightVariable([3,3,64,64])
    # b8 = biasVariable([64])
    # conv8 = tf.nn.relu(conv2d(drop7, W8) + b8)
    # # pool8 = maxPool(conv8)
    # pool8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop8 = dropout(pool8, keep_prob_5)
    #
    # # 第9层
    # W9 = weightVariable([3,3,64,64])
    # b9 = biasVariable([64])
    # conv9 = tf.nn.relu(conv2d(drop8, W9) + b9)
    # # pool9 = maxPool(conv9)
    # pool9 = tf.nn.max_pool(conv9, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop9 = dropout(pool9, keep_prob_5)
    #
    # # 第10层
    # W10 = weightVariable([3,3,64,64])
    # b10 = biasVariable([64])
    # conv10 = tf.nn.relu(conv2d(drop9, W10) + b10)
    # # pool10 = maxPool(conv10)
    # pool10 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop10 = dropout(pool10, keep_prob_5)
    #
    # # 第11层
    # W11 = weightVariable([3,3,64,64])
    # b11 = biasVariable([64])
    # conv11 = tf.nn.relu(conv2d(drop10, W11) + b11)
    # # pool11 = maxPool(conv11)
    # pool11 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop11 = dropout(pool11, keep_prob_5)
    #
    # # 第12层
    # W12 = weightVariable([3,3,64,64])
    # b12 = biasVariable([64])
    # conv12 = tf.nn.relu(conv2d(drop11, W12) + b12)
    # # pool12 = maxPool(conv12)
    # pool12 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop12 = dropout(pool12, keep_prob_5)
    #
    # # 第13层
    # W13 = weightVariable([3,3,64,64])
    # b13 = biasVariable([64])
    # conv13 = tf.nn.relu(conv2d(drop12, W13) + b13)
    # # pool13 = maxPool(conv13)
    # pool13 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop13 = dropout(pool13, keep_prob_5)
    #
    # # 第14层
    # W14 = weightVariable([3,3,64,64])
    # b14 = biasVariable([64])
    # conv14 = tf.nn.relu(conv2d(drop13, W14) + b14)
    # # pool14 = maxPool(conv14)
    # pool14 = tf.nn.max_pool(conv14, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop14 = dropout(pool14, keep_prob_5)
    #
    # # 第15层
    # W15 = weightVariable([3,3,64,64])
    # b15 = biasVariable([64])
    # conv15 = tf.nn.relu(conv2d(drop14, W15) + b15)
    # # pool15 = maxPool(conv15)
    # pool15 = tf.nn.max_pool(conv15, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop15 = dropout(pool15, keep_prob_5)
    #
    # # 第16层
    # W16 = weightVariable([3,3,64,64])
    # b16 = biasVariable([64])
    # conv16 = tf.nn.relu(conv2d(drop15, W16) + b16)
    # # pool16 = maxPool(conv16)
    # pool16 = tf.nn.max_pool(conv16, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop16 = dropout(pool16, keep_prob_5)
    #
    # # 第17层
    # W17 = weightVariable([3,3,64,64])
    # b17 = biasVariable([64])
    # conv17 = tf.nn.relu(conv2d(drop16, W17) + b17)
    # # pool17 = maxPool(conv17)
    # pool17 = tf.nn.max_pool(conv17, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop17 = dropout(pool17, keep_prob_5)
    #
    # # 第18层
    # W18 = weightVariable([3,3,64,64])
    # b18 = biasVariable([64])
    # conv18 = tf.nn.relu(conv2d(drop17, W18) + b18)
    # pool18 = maxPool(conv18)
    # # pool18 = tf.nn.max_pool(conv18, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop18 = dropout(pool18, keep_prob_5)
    #
    # # 第19层
    # W19 = weightVariable([3,3,64,64])
    # b19 = biasVariable([64])
    # conv19 = tf.nn.relu(conv2d(drop18, W19) + b19)
    # pool19 = maxPool(conv19)
    # # pool19 = tf.nn.max_pool(conv19, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop19 = dropout(pool19, keep_prob_5)
    #
    # # 第20层
    # W20 = weightVariable([3,3,64,64])
    # b20 = biasVariable([64])
    # conv20 = tf.nn.relu(conv2d(drop19, W20) + b20)
    # pool20 = maxPool(conv20)
    # # pool20 = tf.nn.max_pool(conv20, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # drop20 = dropout(pool20, keep_prob_5)

    '''# # # 全连接层
    # # Wf = weightVariable([8 * 8 * 64, 512])
    # # bf = biasVariable([512])
    # # drop_flat = tf.reshape(drop6, [-1, 8 * 8 * 64])   #
    # # dense = tf.nn.relu(tf.matmul(drop_flat, Wf) + bf) #
    # # dropf = dropout(dense, keep_prob_75)'''

    # 全连接层1
    # 该全连接层输出的大小
    outf1 = 64
    Wf1 = weightVariable([8 * 8 * 256, outf1])
    bf1 = biasVariable([outf1])
    drop_flat1 = tf.reshape(drop5, [-1, 8 * 8 * 256])
    dense1 = tf.nn.relu(tf.matmul(drop_flat1, Wf1) + bf1)
    dropf = dropout(dense1, keep_prob_75)

    # 全连接层2
    # outf2 = 256
    # Wf2 = weightVariable([outf1, outf2])
    # bf2 = biasVariable([outf2])
    # drop_flat2 = tf.reshape(dropf1, [-1, outf1])
    # dense2 = tf.nn.relu(tf.matmul(drop_flat2, Wf2) + bf2)
    # dropf = dropout(dense2, keep_prob_75)

    # 输出层
    Wout = weightVariable([outf1, len(faces)])
    bout = biasVariable([len(faces)])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

# 获取预测器
output = cnnLayer()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
# saver.restore(sess, tf.train.latest_checkpoint('.'))
saver.restore(sess, model_path)


def is_whose_face(image):
    res = sess.run(predict, feed_dict={x: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})
    # 返回下标，与faces_path对应
    return int(res)

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

for test_faces in test_path:
    files = os.listdir(test_faces)
    files_num = len(files)

    cant_get_face = 0

    for index in range(files_num):   # 小数据集全部提取
        img = cv2.imread(test_faces+"/"+str(files[index]))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_image, 1)
        if not len(dets):
            cant_get_face += 1

            cv2.imshow('img', img)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1, x2:y2]
            # 调整图片的尺寸
            face = cv2.resize(face, (size, size))
            whose_face = faces[is_whose_face(face)]

            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            cv2.putText(img, whose_face, (x2, y1), 2, 1.0, (255, 255, 255), 1)
            cv2.imshow('image', img)
            key = cv2.waitKey(30) & 0xff
            dlib.hit_enter_to_continue()

