import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

X = tf.Variable([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
], dtype=tf.float32)

print('-- Part 1 ----------------------------------')
print(X)

print('-- Part 2 ----------------------------------')
print(tf.size(X))
print(tf.shape(X))
print(tf.size(tf.shape(X)))

print('-- X[0] -------------------')
print(X[0].numpy())
print('---------------')
print('-- X[0, 0] -------------------')
print(X[0,0].numpy())
print('---------------')
print('-- X[0, 0, 0] -------------------')
print(X[0,0,0].numpy())
print('---------------')

