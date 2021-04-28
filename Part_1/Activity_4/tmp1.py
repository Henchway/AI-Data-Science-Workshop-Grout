import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1, 2, 3, 4])
b = tf.constant([5, 6, 7, 9])

c = tf.math.add(a, b)
print(a)
print(b)
print(c)
