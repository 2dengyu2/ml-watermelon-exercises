#!/usr/bin/env python
# _*_coding:utf-8_*_
# 标准BP算法
# @Time    : 2022/11/12 16:14
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : standard_bp.py
# @Software: PyCharm
import math
import random
import json
from pandas import DataFrame

from dataset.InitDataset import init_dataset, RESULT_ATTR

learning_rate = 0.1


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class BackPropagationNeuralNetwork(object):
    D = []
    n = 0.1
    input_layer = []
    hidden_layer = []
    output_layer = []
    # 输入层=>隐藏层
    connection_i_h = []
    # 隐藏层=>输出层
    connection_h_o = []
    epoch = 10
    _attr_list = []
    _hidden_layer_node_count = 2
    _seed = 5.5

    def __init__(self, dataset, attr_list, hidden_layer_node_count, learning_rate, epoch, seed=5.5):
        # 初始化数据集及学习率
        self.n = learning_rate
        self._attr_list = attr_list
        self._hidden_layer_node_count = hidden_layer_node_count
        self.epoch = epoch
        self._seed = seed
        # 清洗数据集
        self.D = self.wash(dataset)
        self._x = self.D[self._attr_list]
        self._y = self.D[RESULT_ATTR]
        self._m = len(self.D)
        # 初始化连接权及各节点阈值
        random.seed(5.5)
        self.input_layer, self.hidden_layer, self.output_layer = self.init_node()
        self.connection_i_h, self.connection_h_o = self.init_connection()
        # output layer 神经元个数
        self._l = len(self.output_layer)
        # hidden layer 神经元个数
        self._q = len(self.hidden_layer)
        # input layer 神经元个数
        self._d = len(self.input_layer)
        self.display()

    def training(self):
        """
        P104 图5.8 误差逆传播算法
        """
        for epoch in range(self.epoch):
            # 对于每个样本的输入x_k、输出y_k
            for k in range(self._m):
                x_k = self._x.iloc[k]
                y_k = self._y.iloc[k]
                # 计算当前样本的输出y_hat_k
                y_hat_k = self.calculate_sample_output_by_5_3()
                # 计算输出层神经元梯度g_j
                g_j = self.calculate_output_layer_gradient_by_5_10()
                # 计算隐层神经元梯度e_h
                e_h = self.calculate_hidden_layer_gradient_by_5_15()
                # 更新连接权w_hj、v_ih与阈值theta_j，gamma_h
                self.update_connect_weight_by_5_11_to_14()

    def update_connect_weight_by_5_11_to_14(self):
        pass

    def calculate_hidden_layer_gradient_by_5_15(self):
        pass

    def calculate_output_layer_gradient_by_5_10(self):
        pass

    def calculate_sample_output_by_5_3(self):
        y_hat_k = []
        for j in range(self._l):
            # sigmoid(输入值-阈值)
            y_hat_k.append(sigmoid(self.beta(j) - self.theta(j)))
        return y_hat_k

    def save(self):
        return None

    def init_connection(self):
        cnt_i = len(self.input_layer)
        cnt_h = len(self.hidden_layer)
        cnt_o = len(self.output_layer)
        _connection_i_h = []
        _connection_h_o = []
        # 输入层=>隐藏层
        for i in range(cnt_i):
            _connection_i_h.append([])
            for h in range(cnt_h):
                _connection_i_h[i].append({'weight': random.random()})
        # 隐藏层=>输出层
        for h in range(cnt_h):
            _connection_h_o.append([])
            for o in range(cnt_o):
                _connection_h_o[h].append({'weight': random.random()})
        return _connection_i_h, _connection_h_o

    def init_node(self):
        _input_layer = []
        _hidden_layer = []
        _output_layer = []
        # 每个参与训练的属性添加一个input node
        for attr in self._attr_list:
            # 属性值，例如脐部、根蒂
            _input_layer.append({'attr_value': attr})
        # 根据隐藏层节点数初始化节点
        for idx in range(self._hidden_layer_node_count):
            _hidden_layer.append({'threshold': random.random()})
        # 根据分类结果数量初始化输出层
        for result in self.D[RESULT_ATTR].unique():
            _output_layer.append({'output': result, 'threshold': random.random()})
        return _input_layer, _hidden_layer, _output_layer

    def wash(self, dataset: DataFrame):
        # 只保留参与计算的字段及结果
        _d = dataset[[*self._attr_list, RESULT_ATTR]]
        _result_dataset = _d.copy()
        # 离散属性转换为数值
        for attr in self._attr_list:
            _map = {}
            _val_list = _d[attr].unique()
            for idx in range(len(_val_list)):
                _map[_val_list[idx]] = idx
            for idx in range(len(_d[attr])):
                _result_dataset.loc[idx, attr] = _map[_d.loc[idx, attr]]
        return _result_dataset

    def display(self):
        _x_max = 700
        _y_max = 500
        _data_i = []
        _data_h = []
        _data_o = []
        data = []
        links = []
        # 绘制各节点
        _len_i = len(self.input_layer)
        _len_h = len(self.hidden_layer)
        _len_o = len(self.output_layer)
        for idx in range(_len_i):
            _data_i.append({'name': self.input_layer[idx]['attr_value'], 'x': _x_max / 4, 'y': _y_max / _len_i * idx})
        for idx in range(_len_h):
            threshold = round(self.hidden_layer[idx]['threshold'], 2)
            _data_h.append({
                'name': 'hidden' + str(idx), 'value': threshold,
                'x': _x_max / 4 * 2, 'y': _y_max / _len_h * idx,
                'label': {'formatter': '\n\n\n\n\n\n{c}'}
            })
        for idx in range(_len_o):
            threshold = round(self.output_layer[idx]['threshold'], 2)
            _data_o.append({
                'name': self.output_layer[idx]['output'], 'value': threshold,
                'x': _x_max / 4 * 3, 'y': _y_max / _len_o * idx,
                'label': {'formatter': '\n\n\n{b}\n\n\n{c}'}
            })
        # 绘制连接
        _len_i = len(self.input_layer)
        _len_h = len(self.hidden_layer)
        _len_o = len(self.output_layer)
        for i in range(_len_i):
            for h in range(_len_h):
                links.append({'source': _data_i[i]['name'], 'target': _data_h[h]['name'],
                              'value': str(round(self.connection_i_h[i][h]['weight'], 2)),
                              'label': {'show': True, 'formatter': '{c}', 'padding': [0, 0, 0, 200]}})
        for h in range(_len_h):
            for o in range(_len_o):
                links.append({'source': _data_h[h]['name'], 'target': _data_o[o]['name'],
                              'value': str(round(self.connection_h_o[h][o]['weight'], 2)),
                              'label': {'show': True, 'formatter': '{c}', 'padding': [0, 0, 0, 200]}})
        with open('./nn.json', 'w', encoding='utf8') as f:
            json.dump({"data": [*_data_i, *_data_h, *_data_o], "links": links}, f, ensure_ascii=False, indent=2)

    def alpha(self, h):
        sum = 0
        for i in range(self._d):
            sum += self.connection_i_h[i][h]['weight'] * self.input_layer[i]['attr_value']
        return sum

    def beta(self, j):
        # beta_j = sum(w_HiddenJ * b_Hidden)
        # 第j个输出神经元的输入=sum(隐藏层输出层的每个连接权*此隐藏层神经元的阈值)
        sum = 0
        for h in range(self._q):
            sum += self.connection_h_o[h][j]['weight'] * self.hidden_layer[h]['threshold']
        return sum

    def theta(self, j):
        # theta_j: 第j个输出层神经元的阈值
        return self.output_layer[j]['threshold']


if __name__ == '__main__':
    D = init_dataset('3.0')
    n = learning_rate
    nn = BackPropagationNeuralNetwork(dataset=D, attr_list=['脐部', '根蒂'], hidden_layer_node_count=2,
                                      learning_rate=n, epoch=100)
    nn.training()
    nn.save()
