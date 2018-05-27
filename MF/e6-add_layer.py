import tensorflow as tf


# 神经层里常见的参数通常有weights、biases和激励函数。
# 定义 神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
