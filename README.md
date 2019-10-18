# Machine-Learing

## Linear Regression - 0912

1. 从文件`ex1_1x.dat`和`ex1_1y.dat`中读取数据并描绘散点图。

2. 为x增添一维后利用全梯度下降求出使损失函数 $J(\theta)$ 最小的参数 $\theta$ 并绘图，结果见`result1.png`。然后对x=3.5和x=7时做预测。

3. 绘制 $\theta_0 - \theta_1 - J$ 的函数图像。

4. 对第二组数据`ex1_2x.dat`与`ex1_2y.dat`中的x矩阵做标准化处理，调整学习率`learning_rate`使每次迭代计算出的损失函数较优，结果见`2_cost.png` 。

## Logistic Regression - 1007

1. 从文件`ex2x.dat`和`ex2y.dat`中读取数据并绘制散点图

2. 利用全梯度下降法求出$\theta$，并且绘制决策边界，画出学习过程中损失函数的迭代走势。

3. 利用牛顿迭代法求出$\theta$，绘制决策边界，画出迭代过程中损失函数的迭代走势，并于梯度下降作比较

## Regularization - 1018

+ 线性回归

    1. 从文件`ex3Linx.dat`和`ex3Liny.dat`中读取数据并绘制散点图

    2. 假定预测函数为5次多项式，做$\lambda$在不同取值下、L2正则化后的拟合图像。

+ 逻辑回归

    1. 从文件`ex3Logx.dat`和`ex3Logy.dat`中读取数据并绘制散点图

    2. feature向量为训练数据每一项的单项式组合，即$x=[1, u, v, u^2, uv, v^2, \cdots, v^6]^T$

    3. 对于不同的$\lambda$应用牛顿下降法求最优解并绘制决策边界。