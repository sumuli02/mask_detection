import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure
import pygad

# number = int(input("what is the MNIST number you want?"))
number = 3
data_num = 100
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
def start_training(masklist, num):
    global spectral_data_list
    #number = 28*28-1
    #masklist = [random.randint(0,number) for i in range(10)]
    #masklist=[]
    #print(masklist)
    masklist = [int(i) for i in masklist]
    predictions = []
    for selected_feature in selected_features:
        selected_feature = np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            if selected_feature[i//28][i%28] == 255:
                selected_feature[i // 28][i % 28] = 0
            else:
                selected_feature[i//28][i%28] = num
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    spectral_data_list = []
    for prediction in predictions:
        sum_prediction = prediction / 100
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)[1600:3000]
        for n,i in enumerate(prediction_result):
            if i <= 0.01:
                prediction_result[n] = 0

            '''
            # specific amplitude
            result = 0
            for i in spectral_data_list:
                result += sum(i[415:420])
            '''
            print(".")
            return prediction_result[988]

results = []
for i in range(50):
    results.append(start_training([i],255))

plt.plot(results,".")
plt.show()

'''
guess = [ 5.790e+02  ,4.320e+02,  4.120e+02 , 6.080e+02 , 8.300e+01, 2.670e+02 , 3.840e+02  ,3.660e+02 , 2.370e+02  ,9.300e+01]
print(start_training(guess))

for i in range(5):
    print(len(sp.signal.find_peaks(spectral_data_list[i])[0]))

for i in spectral_data_list[:10]:
    plt.plot(i)
plt.show()
'''

'''
for i in spectral_data_list[:10]:
    plt.plot(i)
plt.show()
'''

'''
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].imshow(selected_features[0].reshape(28,28))
axes[0].set_title("Digit")

axes[1].imshow(predictions[0])
axes[1].set_title("Predicted")

plt.show()
'''