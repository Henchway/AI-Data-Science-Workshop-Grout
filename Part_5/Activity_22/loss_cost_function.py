# -------------------------------------------------------------------
# Loss and cost function for mean squared errors
# -------------------------------------------------------------------

import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Data set values (x and y) and predicted values (y)
# ------------------------------------------------------

x = tf.constant([0., 2., 4., 6., 8., 10.])
y_actual = tf.constant([3., 2., 8., 8., 9., 15.])
y_predicted = tf.constant([2., 4., 6., 8., 10., 12.])

number_of_datapoints = tf.constant(tf.size(x))

print('----------------------------------------------------')
print('-- Dataset and predicted values')
print('----------------------------------------------------')
print(x)
print(y_actual)
print(y_predicted)
print(number_of_datapoints)

# ------------------------------------------------------
# Use TensorFlow maths to find loss and cost functions
# ------------------------------------------------------

print('----------------------------------------------------')
print('-- TensorFlow maths loss function and cost functions')
print('----------------------------------------------------')

y_delta = tf.math.subtract(y_actual, y_predicted)
y_delta_squared = tf.math.square(y_delta)

loss_functions = y_delta_squared
cost_function = tf.math.divide((tf.math.reduce_sum(loss_functions)), tf.cast(number_of_datapoints, dtype=tf.float32))

print(loss_functions)
print(cost_function)

# ------------------------------------------------------
# Use Keras to find loss
# ------------------------------------------------------

mse_loss_keras = tf.keras.losses.MeanSquaredError()
loss_keras = mse_loss_keras(y_actual, y_predicted)

print('----------------------------------------------------')
print('-- Keras loss')
print('----------------------------------------------------')
print(loss_keras)
print('----------------------------------------------------')

# ------------------------------------------------------
# Plot the values
# ------------------------------------------------------

plt.plot(x, y_actual, 'k-o')
plt.plot(x, y_predicted, 'g-o')
plt.plot(x, y_delta, 'r-o')
plt.plot(x, y_delta_squared, 'b-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -------------------------------------------------------------------