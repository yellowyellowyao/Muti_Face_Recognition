# coding=utf-8
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys

# 载入预测器模型，默认选择最新的预测器
# model_path = './train_faces.model-7399'
model_path = tf.train.latest_checkpoint('.')

# 预测时网络要求与训练时网络结构一致!!!
# 若载入之前的预测器，请保持网络结构一致。
from train_CNN import cnnLayer

test_faces1 = r"path1"
test_faces2 = r"path2"

test_path = [test_faces1,
             test_faces2]

faces = ["people1",
         "people2",
         "people3"]

size = 64

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, len(faces)])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# 获取预测器
output = cnnLayer()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
# saver.restore(sess, tf.train.latest_checkpoint('.'))
saver.restore(sess, model_path)


def whose_face(image):
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
            whose_face = faces[whose_face(face)]

            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            cv2.putText(img, whose_face, (x2, y1), 2, 1.0, (255, 255, 255), 1)
            cv2.imshow('image', img)
            key = cv2.waitKey(30) & 0xff
            dlib.hit_enter_to_continue()

