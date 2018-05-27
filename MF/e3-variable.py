import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量
one = tf.constant(1)

# 定义加法步骤
new_value = tf.add(state, one)

# 将state更新成new_value
update = tf.assign(state, new_value)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

