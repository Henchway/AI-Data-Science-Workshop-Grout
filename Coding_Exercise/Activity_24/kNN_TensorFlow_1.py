# ------------------------------------------------------------------
# Filename:    kNN_TensorFlow_1.py
# ------------------------------------------------------------------
# File description:
# Python and TensorFlow image classification using the MNIST dataset.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import argparse
import tensorflow as tf
import time
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ------------------------------------------------------
# Global variables
# ------------------------------------------------------

k_value_tf = tf.constant(3)


# ------------------------------------------------------
# def create_data_points()
# ------------------------------------------------------
# Create the data for two clusters (cluster 0 and cluster 1)
# Data points in cluster 0 belong to class 0 and data points in
# cluster 1 belong to class 1.
# ------------------------------------------------------
# x is the data point in the cluster and class_value is the class number
# Cluster : a cluster of data point values.
# Class   : the label of the class that the data point belongs to.
# ------------------------------------------------------

def create_data_points():
    """Creates random datapoints to be used as the data to which the programs fits the to be examined datapoint to.

    Returns:
        (([(float, float)], [float], [(float, float)], [float])): Returns a tuple with 4 array. Wheres the first is the class0 (x,y) cooridnates, second is array with the amount of class0 points, the third is the class1 (x,y) cooridnates, the forth is an array with the amount of class1 points.
    """
    print('-- Creating the data points')

    # Cluster 0 data points (x0) / Class 0 label (class_value0 = 0)
    num_points_cluster0 = 100
    mu0 = [-0.5, 5]
    covar0 = [[1.5, 0], [0, 1]]
    x0 = np.random.multivariate_normal(mu0, covar0, num_points_cluster0)
    class_value0 = np.zeros(num_points_cluster0)

    # Cluster 1 data points (x1) / Class 1 label (class_value1= 1)
    num_points_cluster1 = 100
    mu1 = [0.5, 0.75]
    covar1 = [[2.5, 1.5], [1.5, 2.5]]
    x1 = np.random.multivariate_normal(mu1, covar1, num_points_cluster1)
    class_value1 = np.ones(num_points_cluster1)

    print('x0              -> %s' % str(x0))
    print('class_value0     -> %s' % str(class_value0))
    print('x1              -> %s' % str(x1))
    print('class_value1    -> %s' % str(class_value1))

    return x0, class_value0, x1, class_value1


# ------------------------------------------------------
# def create_test_point_to_classify()
# ------------------------------------------------------

def create_test_point_to_classify():
    print('-- Creating a test point to classify')

    data_point = np.array([((np.random.random_sample() * 10) - 5), ((np.random.random_sample() * 10) - 3)])

    data_point_tf = tf.constant(data_point)

    return data_point, data_point_tf


# -------------------------------------------------------------------
# get_label(preds)
# -------------------------------------------------------------------

def get_label(preds):
    """Return which class the examined datapoint belongs to.

    Args:
        preds (tf.Tensor: shape=(k,), dtype=float64): An array containing the k neighrest neighbours classes.

    Returns:
        (tf.Tensor: shape=(), dtype=int64): The class to whom the examined datapoint belongs to.
    """
    print('-- Obtaining the class label')

    counts = tf.math.bincount(tf.dtypes.cast(preds, tf.int32))
    arg_max_count = tf.argmax(counts)

    print('preds       -> %s' % str(preds))
    print('counts      -> %s' % str(counts))
    print('arg_max_count -> %s' % str(arg_max_count))

    return arg_max_count


# -------------------------------------------------------------------
# def predict_class(xt, ct, dt, kt)
# -------------------------------------------------------------------

def predict_class(xt, ct, dt, kt):
    """This function calls the kNN algorithm.

    Args:
        xt (tf.Tensor: shape=(n, 2), dtype=float64): An array containing n points that should be used for fitting.
        ct (tf.Tensor: shape=(n,), dtype=float64):   An array that describes which datapoints is in which class.
        dt (tf.Tensor: shape=(2,), dtype=float64):   The datapoint that shell be examined.
        kt (tf.Tensor: shape=(), dtype=int32):       How many neighbours should be considered.

    Returns:
        (tf.Tensor: shape=(kt,), dtype=float64): An array with the classes of the neighrest k neibours.
    """
    print('-- Predicting the class membership')

    neg_one = tf.constant(-1.0, dtype=tf.float64)
    distance = tf.reduce_sum(tf.abs(tf.subtract(xt, dt)), 1)

    print(neg_one)
    print(distance)

    neg_distance = tf.math.scalar_mul(neg_one, distance)
    # val, val_index = tf.nn.top_k(neg_distance, kt)
    val, val_index = tf.math.top_k(neg_distance, kt)
    cp = tf.gather(ct, val_index)

    print('neg_one      -> %s' % str(neg_one))
    print('distance     -> %s' % str(distance))
    print('neg_distance -> %s' % str(neg_distance))
    print('val          -> %s' % str(val))
    print('val_index    -> %s' % str(val_index))
    print('cp           -> %s' % str(cp))

    return cp


# -------------------------------------------------------------------
# def plot_results(x0, x1, data_point, class_value)
# -------------------------------------------------------------------

def plot_results(x0, x1, data_point, class_value):
    """Plots all the datapoints and the given datapoint, colorcoding them by class and adds a legend.

    Args:
        x0 ([(float, float)]): Points of the class1 members.
        x1 ([(float, float)]): Points of the class2 members.
        data_point ((x, y)):   The classified datapoint.
        class_value (string):  The class to whom the data_point belongs to.
    """
    print('-- Plotting the results')

    plt.style.use('default')

    plt.plot(x0[:, 0], x0[:, 1], 'ro', label='class 0')
    plt.plot(x1[:, 0], x1[:, 1], 'bo', label='class 1')
    plt.plot(data_point[0], data_point[1], 'g', marker='D', markersize=10, label='Test data point')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Simple data point classification: Prediction is class %s' % class_value)
    plt.xlabel('Data x-value')
    plt.ylabel('Data y-value')

    plt.show()


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main(datapoint_x, datapoint_y, k_neighbors):
    """This function predicts where the (x, y) points should be classified to in regards to k neighbours.

    Args:
        datapoint_x (float): x coordinate of the point.
        datapoint_y (float): y coordinate of the point.
        k_neighbors (int):   For how many neighbours should the kNN search for.
    """
    # ------------------------------------------------------
    # -- Start of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')
    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 1. Create the data points in each cluster (x0, class_value0, x1, class_value1)
    # 2. Create data point to classify (data_point, data_point_tf)
    # 3. Combine all cluster values into combined lists (x & class_value)
    # 4. Convert (x & class_value) values to TensorFlow constants (x_tf & class_value_tf)
    # ------------------------------------------------------

    (x0, class_value0, x1, class_value1) = create_data_points()
    (data_point, data_point_tf) = create_test_point_to_classify()

    x = np.vstack((x0, x1))
    class_value = np.hstack((class_value0, class_value1))

    x_tf = tf.constant(x)
    class_value_tf = tf.constant(class_value)

    print('x_tf -> %s' % str(x_tf))
    print('class_value_tf   -> %s' % str(class_value_tf))
    print('x                -> %s' % str(x))
    print('class_value      -> %s' % str(class_value))

    # ------------------------------------------------------
    # Run TensorFlow to predict the classification of data point and
    # print the predicted class using nearest 'k_value' data points.
    # ------------------------------------------------------

    pred = predict_class(x_tf, class_value_tf, data_point_tf, k_value_tf)
    class_value_index = pred
    class_value = get_label(class_value_index)

    print(pred)
    print(class_value_index)
    print(class_value)

    print('\n-----------------------------------------------------------')
    print('-- Prediction: data point %s is in class %s' % (str(data_point), class_value))
    print('-----------------------------------------------------------\n')

    # ------------------------------------------------------
    # Plot the data points
    # ------------------------------------------------------

    plot_results(x0, x1, data_point, class_value)

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

def require_input():
    """Asks the user for x, y, k values.

    Returns: (x, y, z): The parsed x, y, z values.
    """
    x = input("x = ")
    y = input("y = ")
    k = input("k = ")
    return x, y, k


def print_menu():
    """Prints a user-friends menu and asks the user for x, y, k values.

    Returns: (x, y, z): The parsed x, y, z values.
    """
    print("Welcome to the kNN algorithm, this algorithm will \n" +
          "calculate the k nearest neighbors of the coordinates you'll\n" +
          "enter in the next step. The x represents the x coordinate, the y represents, \n" +
          "the y coordinate and the k represents the amount of neighbors to check.\n" +
          "The x and y can be of type integer or float, the k value must be of type integer.\n")

    print("Please enter x, y and k:")
    x, y, k = require_input()

    """Checks that the given inputs are valid types"""
    while type(x) not in (int, float) or type(y) not in (int, float) or type(k) != int:
        if "C" in (x, y, k):
            sys.exit("Program aborted.")
        try:
            x = float(x)
            y = float(y)
            k = int(k)
        except:
            print("Unable to read the numbers you entered, please try again or enter C to cancel:")
            x, y, k = require_input()
    return x, y, k


if __name__ == '__main__':
    """
    On application startup, first we check if any parameters are given. If so, we'll try to parse them.
    If no arguments are given, the program starts a menu which will ask the user for the necessary inputs.
    """
    argsParser = argparse.ArgumentParser(description='k-nearest neighbours with tensorflow')
    argsParser.add_argument("-x", metavar='<float>', type=float, required=False, help="The x value of x*a+y=z")
    argsParser.add_argument("-y", metavar='<float>', type=float, required=False, help="The y value of x*a+y=z")
    argsParser.add_argument("-k", metavar='<integer>', type=int, required=False, help="Amount of neighbors to check.")
    args = argsParser.parse_args()

    argCnt = 0
    for arg in vars(args):
        argCnt += 0 if getattr(args, arg) is None else 1

    """If the parameters given are not equal to 3 (corresponding to x, y, k), the program starts the menu, otherwise the program proceeds to execute the main function."""
    if (argCnt == 3):
        datapoint_x, datapoint_y, k_neighbors = print_menu()
    else:
        datapoint_x = args.x
        datapoint_y = args.y
        datapoint_k = args.k

    main(datapoint_x=datapoint_x, datapoint_y=datapoint_y, k_neighbors=k_neighbors)

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
