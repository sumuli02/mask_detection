from sklearn import datasets
import numpy as np
import cv2
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_num = 19
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array([data[0][data_num]])
gray_image = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]
selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = [int(i) for i in selected_label[1]][data_num]
print("label"+ str(selected_label))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")
prediction = loaded_model.predict(np.array(gray_image).reshape(1,28,28,1), verbose=0).reshape(64, 64)


# show the image
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].imshow(gray_image[0].reshape(28,28))
axes[0].set_title("Digit")

axes[1].imshow(prediction)
axes[1].set_title("Predicted")

plt.show()