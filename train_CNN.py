# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from time import time

time_start = time()

train_faces1 = r"path1"
train_faces2 = r"path2"
train_faces3 = r"path3"

train_path = [train_faces1,
              train_faces2,
              train_faces3]

faces = ["people1",
         "people2",
         "people3"]


# 图片块，每次取128张图片
batch_size = 128
size = 64
learn_rate = 0.0001

imgs = []
labs = []


# 需要填充的大小
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


# 填充图片边缘，并记录图片及对应路径
def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # print(img.size) 12288 = 64 * 64 * 3
            # print(img.shape) (64,64,3)
            # print(path)
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)


for face_path in train_path:
    readData(face_path)

# 将图片数据与标签转换成数组
imgs = np.array(imgs)
encodes = []
for lab in labs:
    encode = [0] * len(faces)
    encode[train_path.index(lab)] = 1
    encodes.append(encode)
labs = np.array(encodes)

def splite_data(imgs, labs):
    # 随机划分测试集与训练集op,
    train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05,
                                                        random_state=random.randint(0, 100), shuffle=True)
    # 参数：图片数据的总数，图片的高、宽、通道
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # print(train_x,test_x)
    # 将数据转换成小于1的数
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
    return train_x, test_x, train_y, test_y

train_x, test_x, train_y, test_y = splite_data(imgs, labs)


num_batch = len(train_x) // batch_size
print('num_batch is {}'.format(num_batch))

x = tf.placeholder(tf.float32, [None, size, size, 3])  # 像素点RGB信息
y_ = tf.placeholder(tf.float32, [None, len(faces)])  # 标签
# 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

keep_prob_5 = tf.placeholder(tf.float32)  # 非全连接层dropout数
keep_prob_75 = tf.placeholder(tf.float32)  # 全连接层dropout数


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


def cnnTrain():
    out = cnnLayer()

    # softmax 归一化处理 使得元素和为1
    # Computes the mean of elements across dimensions of a tensor.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    # 数据保存器的初始化
    saver = tf.train.Saver()

    # 如果有GPU，设置占用显存比
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        print("CNN layer is {}".format(out))

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        step = 1
        for shuffle_number in range(100):
            train_x, test_x, train_y, test_y = splite_data(imgs, labs)
            for n in range(100):
                # 每次取128(batch_size)张图片
                for i in range(num_batch):
                    batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                    batch_y = train_y[i * batch_size: (i + 1) * batch_size]

                    # 开始训练数据，同时训练三个变量，返回三个数据
                    _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                                feed_dict={x: batch_x, y_: batch_y, keep_prob_5: 0.5,
                                                           keep_prob_75: 0.75})
                    summary_writer.add_summary(summary, step)
                    # 打印损失
                    # print(step, loss)
                    if step % 10 == 0:
                        print(step, loss)

                    if (step + 1) % 100 == 0:
                        # 获取测试数据的准确率
                        acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        loss = cross_entropy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        acc2 = accuracy.eval({x: batch_x, y_: batch_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        loss2 = cross_entropy.eval({x: batch_x, y_: batch_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        time_end = time()
                        print("{} train steps ".format(step))
                        print("     valid accuracy rate is {:.2%},and loss is {:.6}".format( acc, loss))
                        print("     train accuracy rate is {:.2%},and loss is {:.6}".format( acc2, loss2))
                        print("Cost time is {}s".format(time_end - time_start))

                        if acc > 0.98 and n > 2:
                            # 已训练编号命名 like  train_faces.model-1000
                            saver.save(sess, './train_faces.model', global_step=step)
                            time_end = time()
                            print("Totally cost time is {:.2}s".format(time_end - time_start))
                            sys.exit(0)
                    step += 1

            acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
            loss = cross_entropy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
            print('\n')
            print("            {} train steps ".format(step))
            print("            Epoch Test  Accuracy rate is {:.2%},and loss is {:.6}".format(acc, loss))
            print('\n')

        saver.save(sess, './train_faces.model', global_step=step)
        print("Totally cost time is {:.2f}s".format(time() - time_start))
        print('accuracy less 0.98, exited!')

cnnTrain()

