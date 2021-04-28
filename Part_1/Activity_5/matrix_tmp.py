import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant(2.0)
b = tf.constant([[1, 2, 3], [4, 5, 6]])
c = tf.constant([[7, 8, 9], [10, 11, 12]])

f = tf.math.multiply(b, c)
g = tf.matmul(b, tf.transpose(c))

print(a)
print(b)
print(c)
print(f)
print(g)
