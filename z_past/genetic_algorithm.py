from deap import base, creator, tools, algorithms
import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp

number = 1
data_num = 2000
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
def complex_function(masklist):
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
                selected_feature[i//28][i%28] = 255
        predictions.append(loaded_model.predict(selected_feature.reshape(1,28,28,1), verbose=0).reshape(64, 64))

    result_temp = 0
    spectral_data_list = []
    for prediction in predictions:
        sum_prediction = prediction / 100
        sum_prediction = sum_prediction.flatten()
        prediction_result = sp.special.softmax(sum_prediction)[1600:3000]

        for n,i in enumerate(prediction_result):
            if i <= 0.01:
                result_temp += 1
        result_temp += sum(prediction_result[919:926]) * 1000
        result_temp += sum(prediction_result[983:995]) * 1000
        return result_temp




# Define the problem: minimize weights that are integers between 0 and 784
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 784)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=40)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("evaluate", complex_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=784, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # Create an initial population
    population = toolbox.population(n=300)

    # Apply the genetic algorithm
    result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True)

    # Get the best individual
    best_individual = tools.selBest(result_population, k=1)[0]
    print("Best individual is:", best_individual)
    print("Fitness:", best_individual.fitness.values[0])


if __name__ == "__main__":
    main()