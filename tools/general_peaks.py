"""
Finding the peaks of hyperspectral bin of random 28x28 pictures
:: Limitation ::
1. peaks > 0.01 is found - can later adjust it to be lower value
2. only selecting bins between index 1600 and 3000
(other bins tend to have very low intensity anyway)
"""
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import os
import json
import time

# if you want debug outputs, DEBUG = 1
DEBUG = 1

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/pretrained_model/model-BOHB-MAE.keras"
loaded_model = tf.keras.models.load_model(dir_path)

inten_list = np.zeros(1400)

# can increase the number of loops
for k in range(3000):
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
            inten_list[i] = 1

    if DEBUG and k % 50 == 49:
        print(f"{time.ctime()} : Loop-{k} completed")

if DEBUG:
    print("Counts for each hyperspectral bins")
    print(list(inten_list))

peaks = []
n = 1600 # index of peaks
for i in inten_list:
    if i > 0:
        peaks.append(n)
    n += 1

if DEBUG:
    plt.plot(np.arange(1600,3000), inten_list, ".")
    plt.show()

with open("peaks_record.json", "w") as f:
    json.dump(peaks, f)