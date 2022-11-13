#!/usr/bin/env python
# _*_coding:utf-8_*_
# 标准BP算法
# @Time    : 2022/11/12 16:14
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : standard_bp.py
# @Software: PyCharm
import random

from pandas import DataFrame

from dataset.InitDataset import init_dataset, RESULT_ATTR

learning_rate = 0.1


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
    _attr_list = []
    _hidden_layer_node_count = 2
    _seed = 5.5

    def __init__(self, dataset, learning_rate, attr_list, hidden_layer_node_count, seed=5.5):
        # 初始化数据集及学习率
        self.n = learning_rate
        self._attr_list = attr_list
        self._hidden_layer_node_count = hidden_layer_node_count
        self._seed = seed
        # 清晰数据集
        self.D = self.wash(dataset)
        # 初始化连接权及各节点阈值
        random.seed(5.5)
        self.input_layer, self.hidden_layer, self.output_layer = self.init_node()
        self.connection_i_h, self.connection_h_o = self.init_connection()

    def training(self):
        return None

    def save(self):
        return None

    def init_connection(self):
        cnt_i = len(self.input_layer)
        cnt_h = len(self.hidden_layer)
        cnt_o = len(self.output_layer)
        _connection_i_h = [[] * cnt_i] * cnt_h
        _connection_h_o = [[] * cnt_h] * cnt_o
        # 输入层=>隐藏层
        for i in range(cnt_i):
            for h in range(cnt_h):
                _connection_i_h[i].append({'weight': random.random()})
        # 隐藏层=>输出层
        for h in range(cnt_h):
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


if __name__ == '__main__':
    D = init_dataset('3.0')
    n = learning_rate
    nn = BackPropagationNeuralNetwork(dataset=D, learning_rate=n, attr_list=['脐部', '根蒂'], hidden_layer_node_count=2)
    nn.training()
    nn.save()
