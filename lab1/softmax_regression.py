# coding=utf-8
import numpy as np


# 返回一个概率分布
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 对每一行找到最大值，保持维度，通过减去最大值，确保数值的稳定性，防止指数爆炸
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 对每一行的指数运算结果求和


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and
    # the objective function value of every iteration and update the theta
    m, n = x.shape  # 获取输入数据 x 的行数和列数，其中 m 是样本数，n 是特征数。

    for i in range(iters):
        chengji = np.dot(x, theta.T)  # 计算输入特征和参数的乘积之和
        fenzi = np.exp(chengji)  # 计算 softmax 函数的分子部分，对线性组合进行指数化

        # 计算 softmax 函数的分母，对分子在第二个轴上进行求和，保持维度。
        fenmu = np.sum(fenzi, axis=1, keepdims=True)
        calculate = fenzi / fenmu  # 计算 softmax 的每个类别的概率
        # 计算梯度
        gradient = -np.dot((y.T - calculate).T, x) / m
        # 使用梯度下降更新参数 theta
        theta -= alpha * gradient
        # 计算目标函数值（交叉熵损失）
        loss = -np.sum(np.log(calculate) * y.T) / m
        # 每次迭代时打印损失值
        print(f'第 {i + 1}/{iters} 次迭代，损失值: {loss}')

    return theta
