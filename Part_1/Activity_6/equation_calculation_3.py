import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf

# total = len(range(0, 11)) * len(tf.range(-5.0, 5.5, 0.5)) * len(tf.range(-10, 10.1, 0.1))

all_a = []
all_b = []
all_x = []
all_y = []

for a in range(0, 11):
    for b in tf.range(-5.0, 5.5, 0.5):
        for x in tf.range(-10, 10.1, 0.1):
            all_a.append(tf.constant(a))
            all_b.append(tf.constant(b))
            all_x.append(tf.constant(x))
            all_y.append(tf.add(tf.multiply(a, x), b))

plt.grid()
plt.title('Plot of y = ax + b')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(all_x,  all_y, 'r')
plt.show()
