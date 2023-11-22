# coding=utf-8
import numpy as np
import struct
import os

from data_process import load_mnist, load_data
from train import train
from evaluate import predict, cal_accuracy

if __name__ == '__main__':
    # initialize the parameters needed
    mnist_dir = "mnist_data/"
    train_data_dir = "train-images-idx3-ubyte"
    train_label_dir = "train-labels-idx1-ubyte"
    test_data_dir = "t10k-images-idx3-ubyte"
    test_label_dir = "t10k-labels-idx1-ubyte"
    k = 10
    iters = 10
    alpha = 1.05

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir,
                                                                     test_data_dir, test_label_dir)
    print("Got data. ")

    # train the classifier
    theta = train(train_images, train_labels, k, iters, alpha)
    print("Finished training. ")

    # evaluate on the testset
    y_predict = predict(test_images, theta)

    # 使用 reshape 转换为二维数组
    y_predict_2d = y_predict.reshape(-1, 1)

    # 输出预测结果的类别概率矩阵
    accuracy = cal_accuracy(y_predict_2d, test_labels)
    print("Finished test. ")
    # 调试输出
    print("预测结果：", y_predict_2d)
    print("实际标签：", test_labels)
    print(f"测试集上的准确率：{accuracy*100:.2f}%")
