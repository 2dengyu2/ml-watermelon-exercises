#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time    : 2022/8/7 16:40
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : InformationEntropy.py
# @Software: PyCharm

import pandas as pd
import math
from pandas import DataFrame
import json

# 是否启用增益率，为False表示使用增益值
IS_ENABLE_GAIN_RATIO = False

# 连续属性
l_continuity_attr = ['密度', '含糖率']
# 排除属性
l_exclude_attr = ['编号']

RESULT_ATTR = '好瓜'


def init_dataset():
    with open('../dataset/watermelon-dataset-3.0.csv', 'r', encoding='utf-8') as f:
        _dataset = pd.read_csv(f)
        _dataset.loc[_dataset[RESULT_ATTR].isin(['是']), RESULT_ATTR] = '好瓜'
        _dataset.loc[_dataset[RESULT_ATTR].isin(['否']), RESULT_ATTR] = '坏瓜'
        # print(_dataset)
        return _dataset


class DecisionTreeNode(object):
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

    # noinspection PyShadowingNames
    def generate_tree(self, dataset: DataFrame, attr_list: list):
        self._dataset = dataset
        # 属性列表
        self._attr_list = attr_list
        # 根节点信息熵
        self._root_ent = self.calculate_ent(dataset)
        self._max_samples_cnt_type = self.get_max_samples_type()
        if self.is_all_same_type():
            self.is_leaf = True
            self.type = self._max_samples_cnt_type
            return

        if len(attr_list) == 0 or self.is_all_same_val():
            self.is_leaf = True
            self.type = self._max_samples_cnt_type
            return

        # 最佳划分属性
        self.optimal_partition_attr = self.choice_optimal_partition_attr()
        # 如果是连续值，可选值手动设置为分界点
        if self.optimal_partition_attr in l_continuity_attr:
            _val_list = [-1, 1]
        else:
            _val_list = list(self._dataset.groupby(self.optimal_partition_attr).groups.keys())
        for optimal_attr_val in _val_list:
            if self.optimal_partition_attr in l_continuity_attr:
                best_val = self.continuity_attr_best_val
                if optimal_attr_val == -1:
                    child_dataset = self._dataset[self._dataset[self.optimal_partition_attr].astype(float) <= best_val]
                    parent_val = '是'
                else:
                    child_dataset = self._dataset[self._dataset[self.optimal_partition_attr].astype(float) > best_val]
                    parent_val = '否'
                # 连续值不移除属性
                child_attr_list = [attr for attr in self._attr_list]
            else:
                parent_val = str(optimal_attr_val)
                child_dataset = self._dataset[self._dataset[self.optimal_partition_attr].isin([optimal_attr_val])]
                child_attr_list = [attr for attr in self._attr_list if attr != self.optimal_partition_attr]
            if len(child_dataset) == 0:
                # TODO 此分支如何走到暂不理解
                child = DecisionTreeNode(is_leaf=True, max_samples_cnt_type=self._max_samples_cnt_type,
                                         parent_val=parent_val)
                self.children.append(child)
                return
            else:
                child = DecisionTreeNode(parent_val=parent_val)
                child.generate_tree(dataset=child_dataset, attr_list=child_attr_list)
                self.children.append(child)

    @staticmethod
    def calculate_ent(children_dataset):
        """
        计算信息熵
        :param children_dataset: 要计算的数据集
        """
        _sum = 0
        _n = len(children_dataset)
        # 求和计算每类的信息熵
        for label, df in children_dataset.groupby(RESULT_ATTR):
            _pk = len(df) / _n
            # 计算
            _sum += _pk * math.log(_pk, 2)
        return _sum * -1

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

    def calculate_gain_ratio(self, attr):
        """
        计算各属性信息增益率
        :param attr: 属性名
        """
        _n = len(self._dataset)
        # 固有值
        _iv = 0
        l_val = list(set(self._dataset[attr].values))
        # 生成样本集在属性attr上的取值列表
        if attr in l_continuity_attr:
            # 如果是连续属性
            # 信息增益值
            _max_gain = 0
            l_divide = self.get_candidate_divide_list(l_val)
            # 遍历所有可能的划分点
            for val in l_divide:
                l_same_val = (self._dataset[self._dataset[attr].apply(lambda x: x <= val)],
                              self._dataset[self._dataset[attr].apply(lambda x: x > val)])
                _sum_gain = 0
                for l_ge_or_lt in l_same_val:
                    _sum_gain += len(l_ge_or_lt) / _n * self.calculate_ent(l_ge_or_lt)
                _sum_gain = self._root_ent - _sum_gain
                if IS_ENABLE_GAIN_RATIO:
                    # 参考 https://arxiv.org/pdf/cs/9603103.pdf 3. Modified Assessment of Continuous Attribute
                    # 此时增益率需要减去log2(N-1)/|D|，N为不重复的值的个数，N-1即划分点的个数，|D|为样本总数
                    _iv = math.log(len(l_divide), 2) / _n
                    _sum_gain -= _iv
                if _sum_gain > _max_gain:
                    _max_gain = _sum_gain
                    self.continuity_attr_best_val = val
            return _max_gain
        else:
            # 如果是离散属性
            # 信息增益值
            _sum_gain = 0
            # 遍历所有可能取值
            for val in l_val:
                # 样本集在attr上相同取值的行
                l_same_val = self._dataset[self._dataset[attr].isin([val])]
                # 计算 gain
                _sum_gain += len(l_same_val) / _n * self.calculate_ent(l_same_val)
                # 计算 iv
                _iv += len(l_same_val) / _n * math.log(len(l_same_val) / _n, 2)
            _gain = self._root_ent - _sum_gain
            if IS_ENABLE_GAIN_RATIO:
                _gain /= (_iv * -1)
            return _gain

    def choice_optimal_partition_attr(self):
        """
        选择最佳根节点划分属性
        :return: 划分属性名称
        """
        optimal_partition_attr = ''
        d_optimal_partition_gain_ratio = 0
        continuity_attr_best_val = 0
        for attr in self._attr_list:
            gain_ratio = self.calculate_gain_ratio(attr)
            if gain_ratio > d_optimal_partition_gain_ratio:
                d_optimal_partition_gain_ratio = gain_ratio
                optimal_partition_attr = attr
                continuity_attr_best_val = self.continuity_attr_best_val
        self.continuity_attr_best_val = continuity_attr_best_val
        return optimal_partition_attr

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
            child_node = tree_to_json(child)
            child_node['value'] = child.parent_val
            children.append(child_node)
    if children is not None and len(children) > 0:
        node_obj['children'] = children
    return node_obj


if __name__ == '__main__':
    dataset = init_dataset()
    attr_list = dataset.columns.values.tolist()
    attr_list.remove(RESULT_ATTR)
    for exclude_attr in l_exclude_attr:
        attr_list.remove(exclude_attr)
    decision_tree = DecisionTreeNode()
    decision_tree.generate_tree(dataset=dataset, attr_list=attr_list)
    # 打印决策树
    tree_root_node = tree_to_json(decision_tree)
    print(json.dumps([tree_root_node], ensure_ascii=False))
