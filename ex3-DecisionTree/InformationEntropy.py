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
l_continuity_attr = ['密度', '含糖率']


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
        # print(dataset)


def get_candidate_divide_list(continuous_sample_list):
    """
    计算连续样本的候选集合T
    :param continuous_sample_list: 连续样本值list
    :return: list
    """
    continuous_sample_list.sort()
    _l_divided = []
    for i in range(len(continuous_sample_list) - 1):
        _l_divided.append((continuous_sample_list[i] + continuous_sample_list[i + 1]) / 2)
    return _l_divided


# noinspection PyShadowingNames
def calculate_gain_ratio(root_ent, attr):
    """
    计算各属性信息增益率
    :param root_ent: 根节点计算出的信息熵
    :param attr: 属性名
    """
    _n = len(full_dataset)
    # 固有值
    _iv = 0
    l_val = list(set(full_dataset[attr].values))
    # 生成样本集在属性attr上的取值列表
    if attr in l_continuity_attr:
        # 信息增益值
        _max_gain = 0
        l_divide = get_candidate_divide_list(l_val)
        # 遍历所有可能的划分点
        for val in l_divide:
            l_same_val = (full_dataset[full_dataset[attr].apply(lambda x: x <= val)],
                          full_dataset[full_dataset[attr].apply(lambda x: x > val)])
            _sum_gain = 0
            for l_ge_or_lt in l_same_val:
                _sum_gain += len(l_ge_or_lt) / _n * calculate_ent(l_ge_or_lt)
            # 参考 https://arxiv.org/pdf/cs/9603103.pdf 3. Modified Assessment of Continuous Attribute
            # 此时增益率需要减去log2(N-1)/|D|，N为不重复的值的个数，N-1即划分点的个数，|D|为样本总数
            _iv = math.log(len(l_divide), 2) / _n
            _sum_gain = root_ent - _sum_gain - _iv
            _max_gain = max(_max_gain, _sum_gain)
        return _max_gain
    else:
        # 信息增益值
        _sum_gain = 0
        # 遍历所有可能取值
        for val in l_val:
            # 样本集在attr上相同取值的行
            l_same_val = full_dataset[full_dataset[attr].isin([val])]
            # 计算 gain
            _sum_gain += len(l_same_val) / _n * calculate_ent(l_same_val)
            # 计算 iv
            _iv += len(l_same_val) / _n * math.log(len(l_same_val) / _n, 2)
        return (root_ent - _sum_gain) / (_iv * -1)


# noinspection PyShadowingNames
def choice_optimal_partition_attr(root_ent, l_attr):
    """
    选择最佳根节点划分属性
    :param root_ent: 根节点Ent
    :param l_attr: 属性列表
    :return: 划分属性名称
    """
    optimal_partition_attr = ''
    d_optimal_partition_gain_ratio = 0
    for attr in l_attr:
        gain_ratio = calculate_gain_ratio(root_ent, attr)
        if gain_ratio > d_optimal_partition_gain_ratio:
            d_optimal_partition_gain_ratio = gain_ratio
            optimal_partition_attr = attr
    return optimal_partition_attr


if __name__ == '__main__':
    init_dataset()
    # 根节点信息熵
    root_ent = calculate_ent(full_dataset)
    # 属性列表
    l_attr = full_dataset.columns.values.tolist()
    l_attr.remove('好瓜')
    # 最佳划分属性
    optimal_partition_attr = choice_optimal_partition_attr(root_ent, l_attr)
    print(optimal_partition_attr)
