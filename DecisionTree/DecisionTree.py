#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time    : 2022/8/13 16:17
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : DecisionTree.py
# @Software: PyCharm
from abc import ABCMeta, abstractmethod

import pandas as pd
from pandas import DataFrame

RESULT_ATTR = '好瓜'

# 连续属性
l_continuity_attr = ['密度', '含糖率']


class DecisionTreeNode(metaclass=ABCMeta):
    """
    决策树节点
    """
    # 此节点所属分类
    type = ''
    # 是否叶节点
    is_leaf = False
    # 最佳划分属性
    optimal_partition_attr = ''
    # 分支节点
    children = []
    # 样本集D中样本数最多的类
    max_samples_cnt_type = ''
    # 父节点取值
    parent_val = ''
    # 当前节点剩余的训练集D
    _dataset = pd.DataFrame()
    # 当前节点可用的属性集A
    _attr_list = []
    # 当前根节点的Ent值
    _root_ent = 0
    # 连续属性最佳分界点
    continuity_attr_best_val = 0

    def __init__(self, is_leaf=False, max_samples_cnt_type='', parent_val=''):
        self._max_samples_cnt_type = max_samples_cnt_type
        self.is_leaf = is_leaf
        self.parent_val = parent_val
        self.children = []
        self.continuity_attr_best_val = 0

    @staticmethod
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

    @abstractmethod
    def choice_optimal_partition_attr(self):
        pass

    @abstractmethod
    # noinspection PyShadowingNames
    def generate_tree(self, dataset: DataFrame, attr_list: list):
        pass

    def is_all_same_type(self):
        """
        判断训练集D中样本都属于同一类别C
        """
        return len(self._dataset.groupby(RESULT_ATTR)) == 1

    def is_all_same_val(self):
        """
        判断训练集D中所有样本的所有属性取值相同
        """
        for attr in self._attr_list:
            if len(self._dataset.groupby(attr)) != 1:
                return False

        return True

    def get_max_samples_type(self):
        """
        D中样本数最多的类，即决定此节点的类型，以求众数的方式实现
        :return: 分类结果
        """
        return self._dataset[RESULT_ATTR].mode().values[0]

    @staticmethod
    def tree_to_json(node):
        node_obj = {}
        if node.is_leaf:
            node_obj['type'] = node.type
            children = None
        else:
            if node.optimal_partition_attr in l_continuity_attr:
                node_obj['attr'] = node.optimal_partition_attr + '≤' + str(node.continuity_attr_best_val) + '?'
            else:
                node_obj['attr'] = node.optimal_partition_attr + '=?'
            children = []
            for child in node.children:
                child_node = node.tree_to_json(child)
                child_node['value'] = child.parent_val
                children.append(child_node)
        if children is not None and len(children) > 0:
            node_obj['children'] = children
        return node_obj
