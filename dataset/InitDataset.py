#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time    : 2022/11/12 16:24
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : InitDataset.py
# @Software: PyCharm
import os

import pandas as pd

RESULT_ATTR = '好瓜'

path = os.path.dirname(os.path.abspath(__file__))


def init_dataset_3_0():
    """
    返回数据集3.0
    :return: 数据集
    """
    with open(os.path.join(path, 'watermelon-dataset-3.0.csv'), 'r', encoding='utf-8') as f:
        _dataset = pd.read_csv(f)
    _dataset.loc[_dataset[RESULT_ATTR].isin(['是']), RESULT_ATTR] = '好'
    _dataset.loc[_dataset[RESULT_ATTR].isin(['否']), RESULT_ATTR] = '不好'
    return _dataset


def init_dataset(version):
    """
    返回指定版本的数据集
    :param version: 版本号，如3.0
    :return: 数据集
    """
    if version == '3.0':
        return init_dataset_3_0()
