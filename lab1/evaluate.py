# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)#执行了矩阵乘法运算，将测试图像数据 test_images 与参数 theta 相乘。这个操作得到了每个测试样本在各个类别上的得分值
    preds = np.argmax(scores, axis=1)
    return preds


def cal_accuracy(y_pred, y):  # y_pred 是模型对测试集的预测结果，y 是测试集的真实标签
    # TODO: Compute the accuracy among the test set and store it in acc
    # 计算准确率
    correct_predictions = np.sum(y_pred == y)  # 预测正确数
    total_samples = len(y)  # 总样本数
    acc = correct_predictions / total_samples
    # 调试输出
    print(f"正确预测数量：{correct_predictions}")
    print(f"总样本数量：{total_samples}")
    return acc
