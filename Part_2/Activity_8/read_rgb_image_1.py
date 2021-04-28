# ------------------------------------------------------------------
# File name       : read_rgb_image_1.py
# ------------------------------------------------------------------
# File description:
# Read RGB image from file and manipulation using TensorFlow
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

# import tensorflow as tf
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2


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
    # print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))
    print('-- Opencv version     : ' + str(cv2.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    # image = cv2.imread('eight.png')
    image = cv2.imread('Gutenberg.jpg')
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA) / 255.0
    gray_not = cv2.bitwise_not(gray)
    resized_gray_not = cv2.resize(gray_not, (28, 28), interpolation=cv2.INTER_AREA) / 255.0

    print(image.shape)
    print(resized_image.shape)
    print(gray.shape)
    print(resized_gray.shape)
    print(gray_not.shape)
    print(resized_gray_not.shape)

    # ------------------------------------------------------

    print('-- Use Matplotlib to display images')

    fig, axs = plt.subplots(1, 6, figsize=(14, 4))
    plt.suptitle('Images')

    axs[0].imshow(image)
    axs[0].set_title('Raw image', fontsize=10)
    axs[0].set_xlabel('x pixel', fontsize=10)
    axs[0].set_ylabel('y pixel', fontsize=10)

    axs[1].imshow(resized_image)
    axs[1].set_title('Resized image', fontsize=10)
    axs[1].set_xlabel('x pixel', fontsize=10)
    axs[1].set_ylabel('y pixel', fontsize=10)

    axs[2].imshow(gray)
    axs[2].set_title('Grayscale image', fontsize=10)
    axs[2].set_xlabel('x pixel', fontsize=10)
    axs[2].set_ylabel('y pixel', fontsize=10)

    axs[3].imshow(resized_gray)
    axs[3].set_title('Resized grayscale image', fontsize=10)
    axs[3].set_xlabel('x pixel', fontsize=10)
    axs[3].set_ylabel('y pixel', fontsize=10)

    axs[4].imshow(gray_not)
    axs[4].set_title('Bitwise not image', fontsize=10)
    axs[4].set_xlabel('x pixel', fontsize=10)
    axs[4].set_ylabel('y pixel', fontsize=10)

    axs[5].imshow(resized_gray_not)
    axs[5].set_title('Resized Bitwise not image', fontsize=10)
    axs[5].set_xlabel('x pixel', fontsize=10)
    axs[5].set_ylabel('y pixel', fontsize=10)

    plt.show()

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
