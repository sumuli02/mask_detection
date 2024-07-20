import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure
import pygad
import time

number = 6
data_num = 2000
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

filter = np.where(selected_label == number)
selected_label, selected_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
## training part
def start_training(masklist):
    global spectral_data_list
    masklist = mis(masklist)
    predictions = []
    for selected_feature in selected_features:
        selected_feature = np.zeros((28,28))#np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            selected_feature[i//28][i%28] = random.randint(0,255)
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    spectral_data_list = []
    for prediction in predictions:
        sum_prediction = prediction / 100
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)[1600:3000]
        for n,i in enumerate(prediction_result):
            if i <= 0.01:
                prediction_result[n] = 0
        spectral_data_list.append(prediction_result)

    spectral_data_list = np.array(spectral_data_list)
    spectral_data_average = np.sum(spectral_data_list, axis=0) / len(spectral_data_list)
    spectral_data_squared_average = np.sum(spectral_data_list ** 2, axis=0) / len(spectral_data_list)
    spectral_data_std = np.sqrt(spectral_data_squared_average - spectral_data_average ** 2)
    result = sum(spectral_data_std)
    print(time.ctime())
    return result

def which(mg):
    l = []
    for n in range(len(mg)):
        g = bin(mg[n])[2:]
        while len(g) < 4:
            g = '0'+ g
        for i in range(4):
            if g[i] == '1':
                l.append(i + n * 4)
    return l

def mis(mg):
    l = []
    for n in range(len(mg)):
        if mg[n] == 1:
            l.append(n)
    return l

def start_training_array(guess):
    l = []
    if len(guess) == 784:
        return start_training(guess)
    for g in guess:
        l.append(start_training(g))
    return np.array(l)

guess = """[0 1 1 0 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 0
 1 1 1 0 0 1 1 0 0 1 1 1 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1 1 0 1 1 0 1 0 1 0
 0 1 1 0 1 1 1 1 1 1 1 1 0 0 0 1 0 1 1 1 1 0 0 1 0 0 1 0 1 0 0 0 1 1 0 1 1
 1 0 1 0 1 0 0 0 0 0 1 1 1 0 0 1 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 0
 1 0 0 1 1 1 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 0 0 1 0 0 1 1 0 0 1 1 1
 0 1 0 1 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 0
 1 0 0 1 1 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0 0 0 1 1 1 1 1
 0 0 1 0 0 0 1 0 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 0 0 1 1 1 0 1 1 1 1 1 0 1 1
 0 1 1 0 1 1 1 1 0 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1
 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 1 1 1 0 0 0 1
 0 1 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1
 1 1 1 1 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 0 1 1 0 0 1 0 0 1 0 0 0 1
 1 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 0 1 0 0 0 0 1 0 1 0 1 0 0 0 0 1
 1 1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 0 1 0 1 0 1 0 0 1 0 0 0 0 0 0 0
 1 0 1 1 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1 1 1 0 1 1 1 1
 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 1 1 0 1 0
 0 0 1 1 0 0 1 0 1 1 0 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 0 1 0 1 1 0 1
 1 0 0 1 1 0 1 0 1 0 1 1 1 0 0 1 0 0 0 0 0 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 0
 1 0 0 0 1 0 0 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1
 1 1 0 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 1
 0 1 0 1 0 1 1 1 1 0 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 0 1 0 1 0
 1 1 0 1 0 1 0]"""
guess = guess.replace("\n","").replace("[","").replace("]","").split(" ")
guess = [int(i) for i in guess]
print(guess)
print(f"final value of result: {start_training(guess)}")
print(sum(guess))

for i in spectral_data_list[:100]:
    plt.plot(i, ".", alpha = 0.3)
plt.show()

print(f"initial value of result: {start_training([])}")

for i in spectral_data_list[:100]:
    plt.plot(i, ".", alpha = 0.3)
plt.show()
