# Linear Regression - 0912

1. 从文件`ex1_1x.dat`和`ex1_1y.dat`中读取数据并描绘散点图

2. 为x增添一维后利用全梯度下降求出使损失函数$J(\theta)$最小的参数$\theta$并绘图，结果见`result1.png`。然后对x=3.5和x=7时做预测

3. 绘制$\theta_0 - \theta_1 - J$的函数图像

4. 对第二组数据`ex1_2x.dat`与`ex1_2y.dat`中的x矩阵做标准化处理，调整学习率`learning_rate`使每次迭代计算出的损失函数较优，结果见`result2.png`

### matplotlib.pyploy 绘图

figure() 设置画布

plot(x, y) 设置数据

xlabel() 设置x坐标标签

ylabel() 设置y坐标标签

show() **展示画布**

Axes3D.plot_surface(x, y, z) 绘制三维坐标图

Axes3D.contour(x, y, z) 绘制等高线图

### numpy 读dat文件

loadtxt()

### numpy 矩阵操作

hstack((x, y)) x矩阵与y矩阵按行拼接

vstack((x, y)) 按列拼接

reshape(-1, 1) 转换为列向量

x.shape 返回tuple表示 (列，行)

dot(x, y) 如果xy是数组则做内积，为矩阵则做矩阵乘法

transpose() 求转置矩阵

std(x, axis) 沿第axis个变量的方向求标准差

mean(x, axis) 沿第axis个变量的方向求平均值
