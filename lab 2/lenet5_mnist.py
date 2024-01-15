import numpy as np
import pickle
import time
from lenet5_model import LeNet5
import struct
import math


# 读取图像和标签数据集
def readDataset(dataset_path):
    (image_path, label_path) = dataset_path
    with open(label_path, "rb") as label_file:
        magic, dataset_size = struct.unpack(">II", label_file.read(8))
        label_dataset = np.fromfile(label_file, dtype=np.int8)
        print('加载数据 %s, 数据量: %d' % (label_path, dataset_size))
    with open(image_path, "rb") as image_file:
        magic, dataset_size, rows, columns = struct.unpack(">IIII", image_file.read(16))
        image_dataset = np.fromfile(image_file, dtype=np.uint8).reshape(len(label_dataset), rows, columns)
        print('加载数据 %s, 数据量: %d' % (image_path, dataset_size))
    return (image_dataset, label_dataset)


# 图像矩阵填充,lenet5所处理的是32x32的大小，但mnist提供的图像是28x28，所以需要填充
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,), (pad,), (pad,), (0,)), "constant", constant_values=(0, 0))
    return X_pad


# 数据集归一化处理 将图像的像素值归一化到 [0, 1] 范围，然后进行零均值化（减去均值并除以标准差）
def normalise(image):
    image -= image.min()
    image = image / image.max()
    image = (image - np.mean(image)) / np.std(image)
    return image


# 生成随机打乱的小批量数据集，减少训练整个数据集的耗时
def random_mini_batches(image, label, mini_batch_size=256, one_batch=False):
    dataset_size = image.shape[0] # 训练样本数
    mini_batches = []
    #  打乱 (image, label)
    permutation = list(np.random.permutation(dataset_size))
    shuffled_image = image[permutation, :, :, :]
    shuffled_label = label[permutation]
    # 提取一个批量
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)
    # 分割 (shuffled_image, shuffled_label)。减去尾部情况。
    complete_minibatches_number = math.floor(
        dataset_size / mini_batch_size)  # 在分区中每个大小为mini_batch_size的小批次的数量
    for k in range(0, complete_minibatches_number):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    # 处理尾部情况（最后的小批次 < mini_batch_size）
    if dataset_size % mini_batch_size != 0:
        mini_batch_image = shuffled_image[complete_minibatches_number * mini_batch_size: dataset_size, :, :, :]
        mini_batch_label = shuffled_label[complete_minibatches_number * mini_batch_size: dataset_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    return mini_batches

# 加载数据集
def load_dataset(test_image_path, test_label_path, train_image_path, train_label_path):
    print('正在加载数据集...')
    train_dataset = (train_image_path, train_label_path)
    test_dataset = (test_image_path, test_label_path)
    # 读取数据
    train_image, train_label = readDataset(train_dataset)
    test_image, test_label = readDataset(test_dataset)
    # 数据预处理
    train_image_normalised_pad = normalise(zero_pad(train_image[:, :, :, np.newaxis], 2))
    test_image_normalised_pad = normalise(zero_pad(test_image[:, :, :, np.newaxis], 2))
    return (train_image_normalised_pad, train_label), (test_image_normalised_pad, test_label)

# 模型训练
def train(model, train_data, test_data, epoches, learning_rate_list, batch_size):
    # 训练循环
    start_time = time.time()
    error_rate_list = []
    for epoch in range(0, epoches):
        print("迭代", epoch + 1, "开始")
        learning_rate = learning_rate_list[epoch]
        # 打印信息
        print("学习率: {}".format(learning_rate))
        print("分组大小: {}".format(batch_size))
        # 循环每个小批次
        start_time_epoch = time.time()
        cost = 0
        mini_batches = random_mini_batches(train_data[0], train_data[1], batch_size)
        print("训练中:")
        for i in range(len(mini_batches)):
            print('训练次数: %d /总数 %d' % (i, len(mini_batches)))
            batch_image, batch_label = mini_batches[i]
            loss = model.forward_propagation(batch_image, batch_label, 'train')
            print(' 损失: %f' % loss)
            cost += loss
            model.back_propagation(learning_rate)
        error_train, _ = model.forward_propagation(train_data[0], train_data[1], 'test')
        error_test, _ = model.forward_propagation(test_data[0], test_data[1], 'test')
        error_rate_list.append([error_train / 60000, error_test / 10000])
        print("训练正确量/总数量", len(train_data[1]) - error_train, "/", len(train_data[1]))
        print("训练正确率:%.2f%%" % ((len(train_data[1]) - error_train) / len(train_data[1]) * 100))
        print("测试正确量/总数量:", len(test_data[1]) - error_test, "/", len(test_data[1]))
        print("测试正确率:%.2f%%" % ((len(test_data[1]) - error_test) / len(test_data[1]) * 100))
        print("迭代", epoch + 1, "结束")
        with open("model_data/lenet5_data_" + str(epoch + 1) + ".pkl", "wb") as output:
            pickle.dump(model.extract_model(), output, pickle.HIGHEST_PROTOCOL)
    error_rate_list = np.array(error_rate_list).T
    print("总用时:", time.time() - start_time, "sec")
    return error_rate_list


def test(model_path, test_data):
    # read model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("测试 {}:".format(model_path))
    errors, predictions = model.forward_propagation(test_data[0], test_data[1], "test")
    print("正确率:", len(predictions) - errors / len(predictions))


test_image_path = "dataset/MNIST/t10k-images-idx3-ubyte"
test_label_path = "dataset/MNIST/t10k-labels-idx1-ubyte"
train_image_path = "dataset/MNIST/train-images-idx3-ubyte"
train_label_path = "dataset/MNIST/train-labels-idx1-ubyte"
#模型的参数
batch_size = 8 #分成8个小数据集
epoches = 1 #训练10轮
learning_rate_list = np.array([5e-2] * 2 + [2e-2] * 3 + [1e-2] * 3 + [5e-3] * 4 + [1e-3] * 4 + [5e-4] * 4)
#选择学习率

#选择模型进行最终的测试
train_data, test_data = load_dataset(test_image_path, test_label_path, train_image_path, train_label_path)
model = LeNet5()
error_rate_list = train(model, train_data, test_data, epoches, learning_rate_list, batch_size)
test("model_data/lenet5_data_" + str(error_rate_list[1].argmin() + 1) + ".pkl", test_data)
