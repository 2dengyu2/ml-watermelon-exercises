# ml-watermelon-exercises

机器学习-西瓜书-习题

## 当前章节

### 公式推导

#### 3.5=>3.7

由(3.5)

$$
\frac{\partial E_{(w,b)}}{\partial w}=2\left(w\sum_{i=1}^{m}x_i^2-\sum_{i=1}^m{(y_i-b)}x_i\right)
$$

推导至(3.7)

$$
w=\frac{\sum_{i=1}^my_i(x_i-\overline x)}{\sum_{i=1}^mx^2-\frac{1}{m}\left(\sum_{i=1}^mx_i\right)^2}
$$

推导过程：

已知(3.8)

$$
b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i)
$$

得到(※)

$$
则b=\frac1m\sum_{i=1}^my_i-\frac1mw\sum_{i=1}^mx_i
$$

由(3.5)

$$
令 \frac{\partial E_{(w,b)}}{\partial w}=0
$$

$$
w\sum_{i=1}^mx_i^2-\sum_{i=1}^mx_iy_i+b\sum_{i=1}^mx_i=0
$$

将(※)带入上式

$$
\begin{equation}
\begin{split}
w\sum_{i=1}^{m}x_i^2-\sum_{i=1}^m{(y_i-b)}x_i&=0\\
w\sum_{i=1}^{m}x_i^2-\sum_{i=1}^my_ix_i+b\sum_{i=1}^mx_i&=0\\
带入(※)消去b\\
w\sum_{i=1}^mx_i^2+(\frac1m\sum_{i=1}^my_i-\frac1mw\sum_{i=1}^mx_i)\sum_{i=1}^mx_i&=\sum_{i=1}^mx_iy_i\\
w\sum_{i=1}^mx_i^2+\frac1m\sum_{i=1}^mx_i\sum_{i=1}^my_i-\frac1mw\left(\sum_{i=1}^mx_i\right)^2&=\sum_{i=1}^mx_iy_i\\
w\left[\sum_{i=1}^mx_i^2-\frac1m\left(\sum_{i=1}^mx_i\right)^2\right]&=\sum_{i=1}^mx_iy_i-\frac1m\sum_{i=1}^mx_i\sum_{i=1}^my_i\\
使用x的均值\overline x=\frac1m\sum_{i=1}^mx_i简化右侧可得\\
w\left[\sum_{i=1}^mx_i^2-\frac1m\left(\sum_{i=1}^mx_i\right)^2\right]&=\sum_{i=1}^mx_iy_i-\overline x\sum_{i=1}^my_i\\
w\left[\sum_{i=1}^mx_i^2-\frac1m\left(\sum_{i=1}^mx_i\right)^2\right]&=\sum_{i=1}^my_i(x_i-\overline x)\\
整理可得(3.7)
\end{split}
\end{equation}
$$

### 数据集

- [西瓜数据集3.0](dataset/watermelon-dataset-3.0.csv)

### 第3章 线性回归