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

number = 2
contrast = 0
data_num = 2000
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
print(f'number of {number}')
print(len(number_label))


filter = np.where(selected_label == contrast)
contrast_label, contrast_features = list(selected_label[filter]), list(np.array(selected_features)[filter])
print(f'number of {contrast}')
print(len(contrast_label))


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
    count = 0
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
        if predictions[81] > 0.01 or predictions[77] > 0.01:
            count += 1
    print(f'This is count {count}')
    count = 0
    # print(predictions)
    pre = predictions.copy()
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
        if predictions[86] > 0.01 or predictions[77] > 0.01:
            count += 1
    print(f'This is count {count}')
    aft=-predictions+pre
    # print(aft)
    # print(time.ctime())
    maxi = max(predictions)
    mini = -min(predictions)
    if (maxi < mini):
        index = np.argmin(predictions)
    else:
        index = np.argmax(predictions)
    print(predictions)
    print('Overall Result')
    print(f'{pre[index]} , {len(number_features)}')
    print(f'{aft[index]} , {len(contrast_features)}')
    print(f'{index} index')
    return -max(mini,maxi)

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
 1 1 0 1 0 1 0]""" #1

guess="""[1 0 0 0 0 0 1 1 0 0 1 0 0 0 0 1 1 0 1 1 0 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 1 0
 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1 1 1 0 1 1 0 1 0
 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 1 1 0 0
 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0 1
 1 1 0 1 0 0 0 1 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0
 1 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 1 1
 0 1 0 1 0 0 0 1 0 0 0 1 0 1 1 0 1 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1
 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 0
 0 1 1 0 0 0 1 0 1 0 0 1 1 1 0 0 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 0 1 0
 1 0 1 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 1 0 1 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 0
 0 0 1 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 0 0 1
 1 0 0 1 0 1 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 1 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0 0 1 0
 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0
 0 1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 0 0 0 1 1 0 1
 0 0 1 1 0 1 0 0 1 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 1 0 0 0
 0 0 0 0 1 1 1 0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 1
 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0
 0 1 0 1 1 0 0]""" #2, 10
guess='''[0 1 0 1 0 1 1 1 0 1 0 0 0 1 1 0 1 0 0 0 1 0 1 1 1 0 1 0 0 1 0 0 1 1 0 1 1
 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0
 0 1 1 1 0 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 1 1 0 1 0 1 0 1 1 0 1 0 0 1 0 0 0 0 1 1 0
 0 0 1 0 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0
 0 0 1 1 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0
 1 0 0 0 0 0 0 0 0 1 1 1 0 1 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0
 0 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1 0
 1 0 0 1 0 0 0 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1
 0 1 1 0 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0 1 0 1
 1 0 0 0 1 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 1 1 1
 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0
 0 1 0 0 0 1 0 0 0 0 1 0 1 1 0 1 0 1 0 1 1 0 0 0 0 1 0 0 1 0 1 1 0 1 0 0 0
 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 0 1 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 0 1 1 1 1 0 0 1 0 0 0 0 1
 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0
 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 0 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0
 0 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1
 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0
 0 1 1 0 1 0 0]''' #6
guess = guess.replace("\n","").replace("[","").replace("]","").split(" ")
guess = [int(i) for i in guess]
print(guess)
print(f"final value of result: {start_training(guess)}")
print(sum(guess))


print(f"initial value of result: {start_training([])}")

