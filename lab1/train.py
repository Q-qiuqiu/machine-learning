# coding=utf-8
import numpy as np

from data_process import data_convert
from softmax_regression import softmax_regression


def train(train_images, train_labels, k, iters=5, alpha=0.5):#迭代次数, 学习率
    m, n = train_images.shape#获取训练图像的形状，其中 m 是样本数，n 是特征数。
    # data processing
    x, y = data_convert(train_images, train_labels, m, k)  # x:[m,n], y:[1,m]

    # Initialize theta.  Use a matrix where each column corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n)  # [k,n]，初始化模型参数 theta，使用随机值初始化
    # do the softmax regression
    theta = softmax_regression(theta, x, y, iters, alpha)#传入参数进行训练学习
    return theta
