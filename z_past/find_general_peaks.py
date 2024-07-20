import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp


dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")


peak_list = np.zeros(1400)
for i in range(1000):
    selected_feature = np.zeros(28*28)
    ran = random.randint(0,28*28-1)
    guess = [random.randint(0,28*28-1) for i in range(ran)]
    for i in guess:
        selected_feature[i] = 255 # random.randint(0,255)

    prediction = loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
    sum_prediction = prediction / 100
    sum_prediction = sum_prediction.flatten()
    prediction_result = sp.special.softmax(sum_prediction)[1600:3000]
    for i, n in enumerate(prediction_result):
        if n > 0.01:
            peak_list[i] = 1

print(list(peak_list))
n = 1600
for i in peak_list:
    if i > 0:
        print(n)
    n += 1