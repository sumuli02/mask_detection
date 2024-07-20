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
number = 7
filter = np.where(selected_label == number)
selected_label, selected_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
peak = 140
peaks = [87, 102, 118, 135, 151, 167, 183, 199]

## 102 result: [629, 831, 209, 692, 689, 295, 707, 137, 649, 14, 362, 724, 353, 584, 75, 22, 255, 117, 376, 704, 449, 660, 364, 114, 229, 546, 478, 471, 282, 106, 430, 353, 9, 497, 668, 737, 831, 274, 528, 689]
## 118 result:
## 151 result: [530, 86, 150, 657, 171, 384, 719, 168, 542, 297, 100, 706, 297, 490, 357, 551, 273, 74, 697, 253, 299, 690, 38, 47, 388, 217, 685, 59, 705, 180, 74, 488, 186, 90, 138, 57, 222, 105, 733, 48]
## 167 result: [442, 533, 547, 394, 398, 320, 409, 370, 104, 95, 178, 525, 594, -15, 102, 101, 294, 394, 186, 62, 488, 618, 653, 129, 198, 495, 598, 280, 806, 636, 533, 497, 716, 462, 686, 323, 721, 571, 359, 761]

## training part
best_prediction = []
def find_best_way_out(num, temp_list, masklist):
    global best_prediction
    value = 1
    index_value = 0
    for n in temp_list:
        masklist[num] = n
        selected_feature = selected_features[0]
        selected_feature = np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            selected_feature[i//28][i%28] = 255
        prediction = loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)

        sum_prediction = skimage.measure.block_reduce(prediction, (4, 4), np.mean) / 500
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)
        spectral_data_list.append(prediction_result)
        if (1 - prediction_result[peak]) < value:
            index_value = n
            value = 1 - prediction_result[peak]
            best_prediction = prediction_result

    return index_value


def start_training(masklist):
    global spectral_data_list
    masklist = [int(i) for i in masklist]
    result = []
    for n, mask in enumerate(masklist):
        temp_list = [mask, mask+1, mask-1, mask+28, mask-28, mask+28+1, mask+28-1, mask-28+1, mask-28-1]
        masklist[n] = find_best_way_out(n, temp_list, masklist)
    return masklist

result = [random.randint(0,28*28-1) for i in range(40)]
for i in range(15):
    print(f"{i}th epoch")
    random.shuffle(result)
    result = start_training(result)
    if i%5 == 0:
        print(f"best value so far {1 - best_prediction[peak]}")

print("this is the result")
print(result)
plt.plot(best_prediction)
plt.plot([peak],[0],".")
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