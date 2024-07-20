import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure
import pygad

number = int(input("what is the MNIST number you want?"))
data_num = 2000
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

print(selected_label)

filter = np.where(selected_label == number)
selected_label, selected_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
peaks = [87, 102, 118, 135, 151, 167, 183, 199]
## training part
def show(masklist):
    global spectral_data_list
    count = 0
    masklist = [int(i) for i in masklist]
    predictions = []
    for selected_feature in selected_features:
        selected_feature = np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            selected_feature[i//28][i%28] = 255
        selected_feature = loaded_model.predict(selected_feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
        selected_feature = selected_feature.flatten()
        selected_feature = sp.special.softmax(selected_feature)[1600:3000]
        selected_feature = np.array(selected_feature)
        peaks = [922,923,924,986,987,989,990,991,992, 993]
        if sum(selected_feature[peaks]) > 0.2:
            count += 1
        # plt.imshow(selected_feature)
        # plt.show()
    return count

guess = [412, 168, 673, 388, 230, 383, 698,  42, 465, 290, 424, 412,   1, 697,
 513, 221,  76, 385, 541, 157, 725, 571, 692, 545, 136, 123, 681, 201,
 608, 108, 309, 541, 769, 563, 621, 404,  97, 699, 134, 539, 637, 596,
 260, 605, 778, 419, 308, 645, 767, 487,  99, 778, 200, 369, 218, 619,
 577, 618,  91,  23, 661, 244, 735,  66, 338,  41, 126, 463, 668, 170,
 605, 261,  63, 246, 618, 225, 783, 206, 459, 255]
print(len(guess))
print(len(selected_features))
print(show(guess))