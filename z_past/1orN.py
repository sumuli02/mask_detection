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
number = 1
contrast_num = 2
data_num = 2000
masklist = [412, 168, 673, 388, 230, 383, 698,  42, 465, 290, 424, 412,   1, 697,
 513, 221,  76, 385, 541, 157, 725, 571, 692, 545, 136, 123, 681, 201,
 608, 108, 309, 541, 769, 563, 621, 404,  97, 699, 134, 539, 637, 596,
 260, 605, 778, 419, 308, 645, 767, 487,  99, 778, 200, 369, 218, 619,
 577, 618,  91,  23, 661, 244, 735,  66, 338,  41, 126, 463, 668, 170,
 605, 261,  63, 246, 618, 225, 783, 206, 459, 255]

data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

print(selected_label)

filter = np.where(selected_label == number)
selected_label_1, selected_features_1 = list(selected_label[filter]), list(np.array(selected_features)[filter])

filter = np.where(selected_label == contrast_num)
selected_label_2, selected_features_2 = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features_1))

print(f"number of selected contrast label (label is {contrast_num}): ")
print(len(selected_features_2))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []

## training part
def start_training_peaks(selected_features, peaks):
    global spectral_data_list, masklist
    predictions = []
    for selected_feature in selected_features:
        selected_feature = np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            if selected_feature[i//28][i%28] == 255:
                selected_feature[i // 28][i % 28] = 0
            else:
                selected_feature[i//28][i%28] = 255
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    result_temp = 0
    spectral_data_list = []
    for prediction in predictions:
        sum_prediction = prediction / 100
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)[1600:3000]

        for p in peaks[:10]:
            p = int(p)
            result_temp += prediction_result[p] * 1000
        for p in peaks[10:]:
            p = int(p)
            result_temp -= prediction_result[p] * 1000
        spectral_data_list.append(prediction_result)

    spectral_data_list = np.array(spectral_data_list)
    print(".")
    result = -result_temp
    return result

def find_best_sorting(random_peaks):
    return start_training_peaks( selected_features_1,random_peaks) - start_training_peaks(selected_features_2,random_peaks)

result = sp.optimize.minimize(
    find_best_sorting,         # The objective function to minimize
    [920, 921,922,923,924, 987,988, 989,990,991, 582, 583, 584, 585, 586, 587, 472, 846, 847, 398  ],            # Initial guess for the weights
    method='L-BFGS-B',          # Using L-BFGS-B method as it's suitable for large problems
    options={'maxiter': 1000}   # Maximum number of iterations
)

# Get the optimized weights
optimized_weights = result.x
print(result.x)

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