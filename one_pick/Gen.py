'''
Optimisation using genetic algorithm

To Do:
Make this into class like pso
'''

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
contrast = 8
data_num = 1000
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
        for i in masklist:
            i *= 200
            i = int(i)
            while i >= 28*28:
                i -= 28*28
            while i < 0:
                i += 28*28
            number_feature[i//28][i%28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = sp.special.softmax(prediction)[1600:3000]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] += 1

    for contrast_feature in contrast_features:
        number_feature = np.array(contrast_feature).reshape(28, 28)
        for i in masklist:
            i *= 200
            i = int(i)
            while i >= 28 * 28:
                i -= 28 * 28
            while i < 0:
                i += 28 * 28
            number_feature[i//28][i%28] = 255
        prediction = loaded_model.predict(number_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64)
        prediction = prediction / 100
        prediction = prediction.flatten()
        prediction = sp.special.softmax(prediction)[1600:3000]
        for i, n in enumerate(prediction):
            if n > 0.01:
                predictions[i] -= 1
    print(predictions)
    max(predictions)
    print(time.ctime())
    result = max([max(predictions), -min(predictions)])
    return result

guess = [random.randint(0,784) for i in range(50)]
num_generations = 15
num_parents_mating = 3

sol_per_pop = 5
num_genes = len(guess)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=start_training,
                       parent_selection_type='sss',
                       crossover_type='single_point',
                       keep_parents=1)

ga_instance.run()

ga_instance.plot_fitness(label=['Obj 1'])

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")