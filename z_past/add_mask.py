import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure

data_num = 2000
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

print(selected_label)
filter = np.where(selected_label == 7)
selected_label, selected_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print("number of selected label (label is 2): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []

## training part
def start_training(masklist):
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
            selected_feature[i//28][i%28] = 255
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    spectral_data_list = []
    for prediction in predictions:
        ### adding all the spatial parts
        # divided by 10000 so that softmax calculation does not reach overflow (max ~ 25000)
        # sum_prediction = np.sum(prediction, axis=0)/10000

        ### using hyperspectra - 2 by 2 binning
        # divided by 500 so that softmax calculation does not reach overflow
        sum_prediction = skimage.measure.block_reduce(prediction, (4, 4), np.mean)/500
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)
        spectral_data_list.append(prediction_result)

    spectral_data_list = np.array(spectral_data_list)
    spectral_data_average = np.sum(spectral_data_list, axis=0) / len(spectral_data_list)
    spectral_data_squared_average = np.sum(spectral_data_list**2, axis=0) / len(spectral_data_list)
    spectral_data_std = np.sqrt(spectral_data_squared_average - spectral_data_average**2 )
    result = sum(spectral_data_std)
    return result

guess = [random.randint(0,28*28-1) for i in range(10)]
print(f"This is the guess for the mask: {guess}")

res = sp.optimize.minimize(start_training, guess)
print(res)

'''
guess = [ 7.660e+02,5.600e+02,5.300e+02,7.290e+02,1.180e+02, 3.150e+02, 2.350e+02, 2.190e+02, 2.910e+02, 3.350e+02]
guess = []
print(start_training(guess))

print(sp.signal.find_peaks(spectral_data_list[0]))

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