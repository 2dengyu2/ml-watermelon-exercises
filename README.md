[TOC]
# ml-watermelon-exercises

机器学习-西瓜书-习题

## 当前章节

### 数据集

- [西瓜数据集3.0](dataset/watermelon-dataset-3.0.csv)

### TODO list

- [ ] `__main__`框架
- [ ] 神经网络的连接权及阈值设计
- [ ] 离散属性值的数值化
- [ ] 初始化连接权及阈值
- [ ] 打印或可视化神经网络
- [ ] sigmoid函数
- [ ] 计算样本输出$\hat{y}_k$
- [ ] 计算输出层梯度$g_j$
- [ ] 计算隐层梯度$e_h$
- [ ] 更新连接权的阈值

## 详细设计

### 数据结构

连接权及阈值input_layer、hidden_layer、output_layer、connection_weight

- input_layer中的节点 $x_i$ 包含以下属性：

  | #    | 名称       | 描述                   |
  | ---- | ---------- | ---------------------- |
  | 1    | attr_value | 属性值，例如脐部、根蒂 |

- hidden_layer中的节点$b_h$包含以下属性：

  | #    | 名称      | 描述 |
  | ---- | --------- | ---- |
  | 1    | threshold | 阈值 |

- output_layer中的节点$y_j$包含以下属性：

  | #    | 名称      | 描述     |
  | ---- | --------- | -------- |
  | 1    | output    | 分类结果 |
  | 2    | threshold | 阈值     |

- connection_weight中的连接$v_{ih}$或$w_{hj}$包含以下属性：

  | #    | 名称   | 描述           |
  | ---- | ------ | -------------- |
  | 1    | from   | 上一层节点idx  |
  | 2    | to     | 下一层节点idx  |
  | 3    | weight | 当前连接的权重 |

