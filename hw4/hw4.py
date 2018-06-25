import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import datetime
import warnings

warnings.filterwarnings('ignore')

from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.model_selection import train_test_split
import utils


def generator():
    ll_data_2g = utils.gongcan_to_ll()
    train_data = utils.ll_to_grid(ll_data_2g)

    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)
    train_data.to_csv("X.csv")

    # 接下来是针对 cnn 进行的预处理
    train_scaled = DataFrame()

    # 归一化
    labels = ['RNCID_', 'CellID_', 'AsuLevel_', 'SignalLevel_', 'RSSI_', 'Latitude_', 'Longitude_']
    for label in labels:
        tmp = DataFrame()

        for i in range(1, 8):
            tmp = pd.concat([tmp, train_data[label + str(i)]], axis=1)

        tmp_index = tmp.columns.tolist()

        tmp = tmp.as_matrix()
        # tmp_scaled = scale(tmp)
        min_max_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        tmp_scaled = min_max_scaler.fit_transform(tmp)
        tmp_scaled = DataFrame(tmp_scaled, columns=tmp_index)
        train_scaled = pd.concat([train_scaled, tmp_scaled], axis=1)

    # train_scaled.to_csv('X_scaled.csv')

    train_scaled = pd.concat([train_scaled, train_data[['IMSI', 'MRTime',
                                                        'Longitude', 'Latitude', 'grid_num']]], axis=1)

    X_ = []
    y_ = []
    for index, row in train_scaled.iterrows():
        y_.append(row['grid_num'])
        x_ = []
        for i in range(1, 8):
            tmp = []
            for label in labels:
                tmp.append(row[label + str(i)])
            x_.append(tmp)

        X_.append(x_)
    # X 是生成好的 7*7 数组，作为 feature
    # y_ 是生成好的 label
    X = np.array(X_)
    y_ = np.array(y_)

    # 对生成好的 label 做 onehot 编码
    y = np.zeros(shape=(y_.shape[0], 13 * 17))
    # print(y.shape)
    for i in range(y_.shape[0]):
        y[i][int(y_[i])] = 1

    return X, y


#########卷积网络会有很多的权重和偏置需要创建，先定义好初始化函数以便复用########
# 给权重制造一些随机噪声打破完全对称（比如截断的正态分布噪声，标准差设为0.1）
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 因为我们要使用ReLU，也给偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


########卷积层、池化层接下来重复使用的，分别定义创建函数########
# tf.nn.conv2d是TensorFlow中的2维卷积函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def main():

    X_ = tf.placeholder(tf.float32, shape=[None, 7, 7])
    y_ = tf.placeholder(tf.float32, shape=[None, 221])
    x_image = tf.reshape(X_, [-1, 7, 7, 1])

    ########设计卷积神经网络########
    # 第一层卷积
    # 卷积核尺寸为3*3,1个颜色通道，32个不同的卷积核
    W_conv1 = weight_variable([3, 3, 1, 32])
    # 用conv2d函数进行卷积操作，加上偏置
    b_conv1 = bias_variable([32])
    # 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 第二层卷积（和第一层大致相同，这一层卷积会提取64种特征）
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 512])
    b_fc1 = bias_variable([512])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([512, 221])
    b_fc2 = bias_variable([221])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 计算 loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # 使用线程加速
    tf.train.start_queue_runners()

    X, y = generator()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for i in range(20000):

        rand_index1 = np.random.choice(y_test.shape[0], size=40)
        rand_x1 = X_train[rand_index1]
        rand_y1 = y_train[rand_index1]
        _, loss_value = sess.run([train_step, cross_entropy], feed_dict={X_: rand_x1, y_: rand_y1, keep_prob: 0.6})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X_: rand_x1, y_: rand_y1, keep_prob: 1.0})
            _, loss_value = sess.run([train_step, cross_entropy], feed_dict={X_: rand_x1, y_: rand_y1, keep_prob: 0.6})
            print("step %d, training accuracy: %.4f, training loss: %.4f" % (i, train_accuracy, loss_value))
        # train_step.run(feed_dict={X_: rand_x1, y_: rand_y1, keep_prob: 0.6})

    test_accuracy = accuracy.eval(feed_dict={X_: X_test, y_: y_test, keep_prob: 1.0})
    _, loss_value = sess.run([train_step, cross_entropy], feed_dict={X_: X_test, y_: y_test, keep_prob: 0.6})
    print("testing accuracy: %.4f, testing loss: %.4f" % (test_accuracy, loss_value))


if __name__ == '__main__':
    main()
