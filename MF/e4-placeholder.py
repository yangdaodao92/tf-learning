import tensorflow as tf

# placeholder 是 Tensorflow 中的占位符，暂时储存变量

# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
output = tf.multiply(input1, input2)

# 穿入数据并计算
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))

