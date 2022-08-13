#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time    : 2022/8/7 16:40
# @Author  : liangliang
# @Email   : 649355204@qq.com
# @File    : GiniIndex.py
# @Software: PyCharm

import pandas as pd
from pandas import DataFrame
import math
import json
from DecisionTree.DecisionTree import DecisionTreeNode

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


class InformationEntropyDecisionTreeNode(DecisionTreeNode):

    def generate_tree(self, dataset: DataFrame, attr_list: list):
        self._dataset = dataset
        # 属性列表
        self._attr_list = attr_list
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
                child = InformationEntropyDecisionTreeNode(is_leaf=True,
                                                           max_samples_cnt_type=self._max_samples_cnt_type,
                                                           parent_val=parent_val)
                self.children.append(child)
                return
            else:
                child = InformationEntropyDecisionTreeNode(parent_val=parent_val)
                child.generate_tree(dataset=child_dataset, attr_list=child_attr_list)
                self.children.append(child)

    def choice_optimal_partition_attr(self):
        """
        选择最佳根节点划分属性
        :return: 划分属性名称
        """
        optimal_partition_attr = ''
        continuity_attr_best_val = 0
        optimal_partition_gini_index = 0
        _is_first = True
        for attr in self._attr_list:
            gini_index = self.calculate_gini_index(attr)
            if _is_first:
                _is_first = False
                optimal_partition_gini_index = gini_index
                optimal_partition_attr = attr
                continuity_attr_best_val = self.continuity_attr_best_val
            elif gini_index < optimal_partition_gini_index:
                optimal_partition_gini_index = gini_index
                optimal_partition_attr = attr
                continuity_attr_best_val = self.continuity_attr_best_val
        self.continuity_attr_best_val = continuity_attr_best_val
        return optimal_partition_attr

    @staticmethod
    def calculate_gini(dataset):
        """
        计算数据集D的基尼值
        :param dataset: 数据集
        :return: 基尼值
        """
        _sum = 0
        _n = len(dataset)
        # 求和计算每类的出现频率
        for label, df in dataset.groupby(RESULT_ATTR):
            _pk = len(df) / _n
            _sum += math.pow(_pk, 2)
        return 1 - _sum

    def calculate_gini_index(self, attr):
        """
        计算各属性基尼指数
        :param attr: 属性名
        """
        _n = len(self._dataset)
        l_val = list(set(self._dataset[attr].values))
        # 生成样本集在属性attr上的取值列表
        if attr in l_continuity_attr:
            # 如果是连续属性
            # 信息增益值
            _min_gini = 0
            _is_first = True
            l_divide = self.get_candidate_divide_list(l_val)
            # 遍历所有可能的划分点
            for val in l_divide:
                l_same_val = (self._dataset[self._dataset[attr].apply(lambda x: x <= val)],
                              self._dataset[self._dataset[attr].apply(lambda x: x > val)])
                gini_index_sum = 0
                for l_ge_or_lt in l_same_val:
                    gini_index_sum += len(l_ge_or_lt) / _n * self.calculate_gini(l_ge_or_lt)
                if _is_first:
                    _is_first = False
                    _min_gini = gini_index_sum
                elif gini_index_sum < _min_gini:
                    _min_gini = gini_index_sum
                    self.continuity_attr_best_val = val
            return _min_gini
        else:
            # 如果是离散属性
            gini_index_sum = 0
            # 遍历所有可能取值
            for val in l_val:
                # 样本集在attr上相同取值的行
                l_same_val = self._dataset[self._dataset[attr].isin([val])]
                gini_index_sum += len(l_same_val) / _n * self.calculate_gini(l_same_val)
            return gini_index_sum


if __name__ == '__main__':
    dataset_all = init_dataset()
    attr_list_all = dataset_all.columns.values.tolist()
    attr_list_all.remove(RESULT_ATTR)
    for exclude_attr in l_exclude_attr:
        attr_list_all.remove(exclude_attr)
    decision_tree = InformationEntropyDecisionTreeNode()
    decision_tree.generate_tree(dataset=dataset_all, attr_list=attr_list_all)
    # 打印决策树
    tree_root_node = decision_tree.tree_to_json(decision_tree)
    print(json.dumps([tree_root_node], ensure_ascii=False))
