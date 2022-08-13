# ml-watermelon-exercises

机器学习-西瓜书-习题

## 当前章节

### 数据集

- [西瓜数据集3.0](dataset/watermelon-dataset-3.0.csv)

### 第4章 决策树

> Chapter 4 Decision Tree

- [决策树基类](DecisionTree)

  抽离了公共方法

- [4.3 信息熵进行划分选择的决策树算法](ex3-DecisionTree)

  > 分类结果

    ``` json
    [
        {
            "attr": "纹理=?",
            "children": [
                {
                    "type": "坏瓜",
                    "value": "模糊"
                },
                {
                    "attr": "密度≤0.3815?",
                    "children": [
                        {
                            "type": "坏瓜",
                            "value": "是"
                        },
                        {
                            "type": "好瓜",
                            "value": "否"
                        }
                    ],
                    "value": "清晰"
                },
                {
                    "attr": "触感=?",
                    "children": [
                        {
                            "type": "坏瓜",
                            "value": "硬滑"
                        },
                        {
                            "type": "好瓜",
                            "value": "软粘"
                        }
                    ],
                    "value": "稍糊"
                }
            ]
        }
    ] 
    ```

- 
