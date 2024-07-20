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

number = 1
contrast = 4
data_num = 2000
guess = """[ 0.87679208 -0.73481098  1.53670588  1.70478593 -2.59177537 -1.14902726
  2.31603998  1.51501377  0.3840084  -0.15697282 -0.15755384 -0.57556456
 -1.86873258  1.71932784 -2.86456222  0.11157975 -3.25273147  0.2084936
  3.01420735  2.41084869 -2.4215791   2.95561824 -1.6262012  -2.7530407
  0.52358053 -0.77114657  4.27462197  3.71542958  2.58397648 -3.70501081]""" #3
guess='''[ 3.84761596 -2.51690551 -1.46249866 -0.35173715 -2.48390037 -3.29974163
 -0.82076817 -1.96089619  3.64010721 -1.8269346   0.25447346  3.53251368
  0.36018428 -0.38054049  0.55947013  1.76700635  3.99303296  1.86396063
  2.00573045 -2.19475188  2.11662501  1.66908468 -3.47002334 -3.03437562
  2.96439564 -0.68098411 -1.40947683  0.0954832  -3.49795877 -2.41451716]''' #4 46
guess='''[-3.42669446 -0.96010866  0.98618763 -3.7398362  -3.32201415 -3.63103942
  3.31286043 -3.96633247  1.20463127 -4.43202721 -1.95834086 -1.93055978
 -2.21270777 -1.71657792  3.38028086  1.13196332 -2.18925867 -0.85047602
 -1.93750473 -1.16723093 -3.30876911 -3.24782534  1.16489472 -0.39925177
  1.32662409 -3.14693604 -2.32179144  2.2959473  -0.21544458 -2.7539518 ]''' #5 58 *100 until here
guess = '''[-0.86494637 -1.07925459 -3.10958367 -3.93036435 -2.60641287  1.02941396
  3.02741147  2.88350228 -2.22229443  3.4077365   2.97610578  1.47650112
  1.35192046  2.40797781 -2.37575591  1.78853058 -2.24662058  1.35106664
 -3.59829894  3.05187341  1.09017856 -1.30576173 -0.41715099 -1.13459905
 -2.09867263  2.79485591 -0.43989031 -3.81598363  3.95372451 -0.46816262
  1.4074118  -4.0194622  -4.14867173 -3.03052549 -1.42169107 -1.76226234
 -2.13106911  3.14975024 -2.86906662 -2.16151003 -2.07815619  2.64219331
 -3.64990542  2.6121398   2.65040471 -0.32802275 -1.53881167 -0.76521296
  2.40534689 -3.62569266]''' #6  3vs9
guess = '''[-0.48402624  0.24919038  0.76672405 -1.62791297  3.79935962  3.15083507
 -0.06751837 -1.77515278  1.44333609 -0.67792112  0.24297072 -3.8008161
 -0.48200407 -0.6641186   3.57033077 -1.85852564 -3.29952605  1.04392487
 -0.24386198 -1.14994961  3.71770289  1.23331943  3.47701444 -1.72534243
 -0.52520619  2.35362446 -3.1012672   3.29987473  3.00600279 -1.08345544
  2.85660307  2.49719457  3.15899059  1.89013454  0.74043715 -2.29766602
 -1.97851397 -4.66023266 -0.6391601   1.74145078 -0.1409313   2.91673711
  2.62042443  4.02728596 -3.39535369 -2.25478391  2.3837541   1.99801499
 -2.13850181 -1.96341787]''' #7 3vs7
guess = '''[ 3.55083205  3.21008944  1.73969241  2.03141999 -1.98351706  0.56869276
 -3.51343429  0.18684117 -1.40918544  3.54580222 -2.07137883 -1.70397848
  1.0675731   1.11504978 -1.71724885  0.31281366 -0.84022223 -2.80328251
 -1.10095699 -3.07185197 -2.8857427  -3.31816888  1.94914645  3.31837807
 -1.72916729 -3.76195692  3.03181375  0.39611137 -1.05475628 -3.32784579
  0.45806262 -1.84806779 -3.23257762 -3.43745485  3.11680266  2.0693399
  0.06587456 -2.02263617  3.12145083 -3.53299977 -0.94936705 -2.2257802
 -3.42403413  3.59565474 -1.83471401 -1.85729026  2.37579491 -1.60176329
 -3.36927189 -3.41223066]''' #8 1vs4
guess_1 = guess.replace("\n","").replace("[","").replace("]","").split(" ")
guess = []
for i in guess_1:
    if i!='':
        guess.append(float(i))
print(guess)
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
print(len(number_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
## training part
def start_training(ga_instance, masklist, solution_idx):
    global spectral_data_list
    predictions = np.zeros(1400)
    result_temp = 0

    for number_feature in number_features:
        number_feature = np.array(number_feature).reshape(28, 28)
        '''for i in masklist:
            i *= 100
            if i >= 28*28:
                i -= 28*28
            i = int(i)
            number_feature[i//28][i%28] = 255'''
        for i in masklist:
            i *= 200
            i = int(i)
            while i >= 28*28:
                i -= 28*28
            while i < 0:
                i += 28*28
            number_feature[i // 28][i % 28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = sp.special.softmax(prediction)[1600:3000]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] += 1
    pre = predictions.copy()
    for contrast_feature in contrast_features:
        number_feature = np.array(contrast_feature).reshape(28, 28)
        '''for i in masklist:
            i *= 100
            if i >= 28*28:
                i -= 28*28
            i = int(i)
            number_feature[i//28][i%28] = 255'''
        for i in masklist:
            i *= 200
            i = int(i)
            while i >= 28*28:
                i -= 28*28
            while i < 0:
                i += 28*28
            number_feature[i // 28][i % 28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = sp.special.softmax(prediction)[1600:3000]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] -= 1
    aft = -predictions + pre
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
    return max(mini,maxi)

print(start_training([],guess,[]))