import tensorflow as tf
import numpy as np

sess = tf.Session()
print(sess.run(tf.zeros([1], tf.int32)))

x_data = np.linspace(-1, 1, 5, dtype=np.float32)[:, np.newaxis]
print(x_data)
print(np.random.normal(0, 0.05, x_data.shape).astype(np.float32))
