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

number = 4
contrast = 6
data_num = 1000
peak = [1625, 1626, 1627, 1628,1629,1688,1689,1690,1691,1692,1816,1817,1818,1819,1820,1821,1882,1883,1884,1885,1946,1947,1948,1949,2008,2009,2010,2011,2012,2013,2014,2015,2016,2074,2075,2076,2077,2137,2138,2139,2140,2141,2142,2201,2202,2203,2204,2205,2206,2265,2266,2267,2268,2269,2329,2330,2331,2332,2333,2334,2394,2395,2396,2397,2398,2457,2458,2459,2460,2461,2462,2519,2520,2521,2522,2523,2524,2525,2526,2584,2585,2586,2587,2588,2589,2590,2591,2592,2593,2650,2651,2652]
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

filter = np.where(selected_label == number)
number_label, number_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

filter = np.where(selected_label == contrast)
contrast_label, contrast_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
## training part
def start_training(masklist):
    global spectral_data_list
    sum_list = sum(masklist)
    masklist = mis(masklist)
    predictions = np.zeros(len(peak))
    for number_feature in number_features:
        number_feature = np.array(number_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            i = int(i)
            number_feature[i//28][i%28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = np.array(sp.special.softmax(prediction))[peak]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] += 1
    for contrast_feature in contrast_features:
        number_feature = np.array(contrast_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            i = int(i)
            number_feature[i//28][i%28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = np.array(sp.special.softmax(prediction))[peak]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] -= 1
    print(predictions)
    print(time.ctime())
    result = -max([-min(predictions), max(predictions)])
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

guess_num = [random.randint(0,784) for i in range(250)]
guess_num = []
guess = []
for i in range(784):
    if i in guess_num:
        guess.append(1)
    else:
        guess.append(0)
print(guess)
print(f"initial value of result: {start_training(guess)}")

import pyswarms as ps
swarm_size, dim = 10, len(guess)
options = {'c1': 1.5, 'c2':1.5, 'w':0.5, 'k':4, 'p':1}

optimizer = ps.discrete.binary.BinaryPSO(swarm_size, dim, options, init_pos=np.array([guess]))

cost, joint_vars = optimizer.optimize(start_training_array, iters=10)

print(cost)
print(list(joint_vars))
print(sum(joint_vars))

