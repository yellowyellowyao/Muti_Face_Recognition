#coding=utf-8
import cv2
import dlib
import os
import sys
import random

# dir_No = 1
output_dir = './faces/001'
size = 64
video_path = r"./video_name"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度与对比度用来训练模型，以增加泛化能力
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
# 数字为设备编号,0为默认摄像头
camera = cv2.VideoCapture(0)

# 也可直接输入视频文件路径
# camera = cv2.VideoCapture(video_path)

# 注意再次写入同一文件夹时更改索引，否则会覆盖文件
index = 1
while True:
    if (index <= 40000):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()

        # 进行人脸转正
        img = dlib.jitter_image(img)
        if not success:
            continue
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.75, 1.25), random.randint(-25, 25))

            face = cv2.resize(face, (size, size))
            # 调整图片对比度与亮度之后有可能就识别不出人脸，此时不存入
            # 倘若存入这部分图片且数目过多，会导致训练时正确率达不到收敛要求
            if not len(detector(face, 1)):
                continue

            cv2.imshow('image', face)
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)

            index += 1

        key = cv2.waitKey(1) & 0xff
        # 键码27是ESC
        if key == 27:
            break
    else:
        print('Finished!')
        break
