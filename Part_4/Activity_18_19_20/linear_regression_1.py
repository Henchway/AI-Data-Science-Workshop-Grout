# ------------------------------------------------------------------
# Filename:    linear_regression_1.py
# ------------------------------------------------------------------
# File description:
# Python and TensorFlow simple linear regression.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ------------------------------------------------------
# Global variables
# ------------------------------------------------------

# dataset = [[0, 11], [0, 8], [10, 20], [10, 23],
#            [2, 12], [4, 10], [8, 15], [9, 19],
#            [5, 11], [5, 10], [6, 14], [7, 15]
#            ]

random_x_values = tf.random.uniform((100,), -10, 10, dtype=tf.dtypes.float32)
random_a_values = tf.random.uniform((100,), -5, 25, dtype=tf.dtypes.float32)
random_b_values = tf.random.uniform((100,), -5, 5, dtype=tf.dtypes.float32)
random_y_values = random_a_values * random_x_values + random_b_values
dataset = tf.stack([random_x_values, random_y_values], axis=1)
no_of_epochs = 10


# ---------------------------------------------------------------
# def train_the_model(x, y)
# ---------------------------------------------------------------

def train_the_model(x, y):
    print('-- Train the model')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(x, y, epochs=no_of_epochs, verbose=0)

    print(model.summary())

    return model


# ---------------------------------------------------------------
# def predict_from_model(model, to_predict)
# ---------------------------------------------------------------

def predict_from_model(model, to_predict):
    print('-- Predict from the trained model')

    predictions = model.predict(to_predict)
    weights = model.get_weights()

    return predictions, weights


# ---------------------------------------------------------------
# def display_details(weights, to_predict, predictions)
# ---------------------------------------------------------------

def display_details(weights, to_predict, predictions):
    print('-- Display details')

    a0 = weights[0]
    b0 = weights[1]

    a = a0[0, 0]
    b = b0[0]

    print('---------------------------------------------------')
    print(weights)

    print('-- a = ' + str(a))
    print('-- b = ' + str(b))

    if a == 0.0:
        a_print = ''
    else:
        a_print = str('%.3f' % a)

    if b == 0.0:
        b_print = ''
    elif b < 0.0:
        b_print = str(' - %.3f' % tf.math.abs(b))
    else:
        b_print = str(' + %.3f' % b)

    print('-- Simple linear equation estimation --------------')
    print('-- Equation is y = ' + a_print + b_print)
    print('---------------------------------------------------')

    print('-- Predicted values of y --------------------------')
    print('x = ' + str(to_predict))
    print('')
    print('y = ' + str(predictions))
    print('---------------------------------------------------')


# ---------------------------------------------------------------
# def plot_data(x, y, weights, to_predict, predictions)
# ---------------------------------------------------------------

def plot_data(x, y, weights, to_predict, predictions):
    print('-- Plot the data')

    a = weights[0][0, 0]
    b = weights[1][0]

    x_predictions = np.arange(
        x[tf.argmin(x)],
        x[tf.argmax(x)],
        0.001)

    y_predictions = tf.math.add(tf.math.multiply(a, x_predictions), b)

    plt.plot(x, y, 'ro')
    plt.plot(x_predictions, y_predictions, 'g')
    plt.plot(to_predict, predictions, 'b*')
    plt.grid()
    plt.title('Simple linear regression in Python and TensorFlow')
    plt.xlabel('x data')
    plt.ylabel('y data')
    plt.show()


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():
    # ------------------------------------------------------
    # -- Start of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')
    print('-- Python version     : ' + str(sys.version))
    print('-- NumPy version      : ' + str(np.__version__))
    print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    training_data = dataset

    # x_train = [row[0] for row in training_data]
    # y_train = [row[1] for row in training_data]

    x_train = random_x_values
    y_train = random_y_values

    model = train_the_model(x_train, y_train)

    to_predict = np.array([10, 11, 12, 13])
    predictions, weights = predict_from_model(model, to_predict)

    print('-- Training dataset --------------------------')
    print(training_data)
    print('-- x -----------------------------------------')
    print(x_train)
    print('-- y -----------------------------------------')
    print(y_train)
    print('-- Model and predictions ---------------------')
    display_details(weights, to_predict, predictions)
    print('----------------------------------------------')

    plot_data(x_train, y_train, weights, to_predict, predictions)

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------ script
# ------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
