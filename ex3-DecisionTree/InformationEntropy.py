#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time    : 2022/8/7 16:40
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : InformationEntropy.py
# @Software: PyCharm

import pandas as pd
import math

full_dataset = pd.DataFrame()


def calculate_ent(dataset):
    """
    计算信息熵
    :param dataset: 要计算的数据集
    """
    _sum = 0
    _n = len(dataset)
    # 求和计算每类的信息熵
    for label, df in dataset.groupby('好瓜'):
        _pk = len(df) / _n
        # 计算
        _sum += _pk * math.log(_pk, 2)
    return _sum * -1


def init_dataset():
    global full_dataset
    with open('../dataset/watermelon-dataset-3.0.csv', 'r', encoding='utf-8') as f:
        full_dataset = pd.read_csv(f)
        # TODO 处理连续值
        # print(dataset)


# noinspection PyShadowingNames
def calculate_gain_ratio(root_ent, attr):
    """
    计算各属性信息增益率
    :param root_ent: 根节点计算出的信息熵
    :param attr: 属性名
    """
    # 信息增益值
    _sum_gain = 0
    _n = len(full_dataset)
    # 固有值
    _iv = 0
    # 生成样本集在属性attr上的取值列表
    l_val = list(set(full_dataset[attr].values))
    for val in l_val:
        # 样本集在attr上相同取值的行
        l_same_val = full_dataset[full_dataset[attr].isin([val])]
        # 计算 gain
        _sum_gain += len(l_same_val) / _n * calculate_ent(l_same_val)
        # 计算 iv
        _iv += len(l_same_val) / _n * math.log(len(l_same_val) / _n, 2)
    return (root_ent - _sum_gain) / (_iv * -1)


if __name__ == '__main__':
    init_dataset()
    # 根节点信息熵
    root_ent = calculate_ent(full_dataset)
    # 属性列表
    l_attr = full_dataset.columns.values.tolist()
    l_attr.remove('好瓜')
    for attr in l_attr:
        gain_ratio = calculate_gain_ratio(root_ent, attr)
        print(gain_ratio)
