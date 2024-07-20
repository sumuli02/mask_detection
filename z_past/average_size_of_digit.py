# finding the average size of a picture which contains the MNIST digit

from sklearn import datasets
import numpy as np
import cv2
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_num = 20000
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
gray_images = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]
total_percentage = 0
total_percentage_squared = 0

for gray_image in gray_images:
    initial_sum = sum(sum(gray_image))

    # change this accordingly
    gray_image = np.delete(gray_image, [0,1,2,3,4,-5,-4,-3,-2,-1], 0)
    gray_image = np.delete(gray_image, [0,1,2,3,4,-5,-4,-3,-2,-1], 1)

    final_sum = sum(sum(gray_image))

    total_percentage += initial_sum/final_sum
    total_percentage_squared += (initial_sum/final_sum)**2

print("occupation probability")
print(total_percentage/data_num * 100)

print("standard deviation")
print(np.sqrt(total_percentage_squared/data_num - (total_percentage/data_num)**2) * 100)



#for debug
'''plt.imshow(gray_image)
plt.show()'''

