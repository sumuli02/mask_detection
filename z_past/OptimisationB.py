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
data_num = 1000
data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                 parser='auto', return_X_y=True)

selected_features = np.array(data[0][:data_num])
selected_features = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in selected_features]

selected_label = datasets.fetch_openml('mnist_784', version=1, as_frame=False,
                                      parser='auto', return_X_y = True)
selected_label = np.array([int(i) for i in selected_label[1]][:data_num])

filter = np.where(selected_label == number)
selected_label, selected_features = list(selected_label[filter]), list(np.array(selected_features)[filter])

print(f"number of selected label (label is {number}): ")
print(len(selected_features))

dir_folder = "/Users/emoon/PycharmProjects/BscTest/pretrained_model"
loaded_model = tf.keras.models.load_model(dir_folder + "/model-BOHB-MAE.keras")

spectral_data_list = []
## training part
def start_training(ga_instance, masklist, solution_idx):
    global spectral_data_list
    predictions = []
    result_temp = 0
    for selected_feature in selected_features:
        selected_feature = np.array(selected_feature).reshape(28, 28)
        for i in masklist:
            if i >= 28*28:
                i -= 28*28
            i = int(i)
            selected_feature[i//28][i%28] = 255
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    spectral_data_list = []
    for prediction in predictions:
        sum_prediction = prediction / 100
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)[1600:3000]
        for n,i in enumerate(prediction_result):
            if i <= 0.01:
                prediction_result[n] = 0
                result_temp += 1
        spectral_data_list.append(prediction_result)

    spectral_data_list = np.array(spectral_data_list)
    spectral_data_average = np.sum(spectral_data_list, axis=0) / len(spectral_data_list)
    spectral_data_squared_average = np.sum(spectral_data_list ** 2, axis=0) / len(spectral_data_list)
    spectral_data_std = np.sqrt(spectral_data_squared_average - spectral_data_average ** 2)
    result = sum(spectral_data_std)
    result_temp *= 0.001


    print(time.ctime())
    return [result, -result_temp]

guess = [random.randint(0,784) for i in range(50)]
num_generations = 10
num_parents_mating = 3

sol_per_pop = 5
num_genes = len(guess)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=start_training,
                       parent_selection_type='nsga2')

ga_instance.run()

ga_instance.plot_fitness(label=['Obj 1', 'Obj 2'])

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")