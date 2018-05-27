import tensorflow as tf
import numpy as np

# 创建数据：模拟了一批可用于训练的数据
# 训练的目的就是为了找到 0.1 和0.3这两个常量
# 0.1 weights 权重， 0.3 biases 偏移量
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 搭建模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

# 计算误差
loss = tf.reduce_mean(tf.square(y-y_data))

# 传播误差
# 梯度下降法传播误差，然后使用optimizer进行参数修正
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#### 以上就完成了整个模型的搭建

# 初始化所有定义的Variable
init = tf.global_variables_initializer()

# 然后创建会话
# 使用session执行init初始化
# 使用session执行每一次的training，逐步提升神经网络的预测准确性
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

