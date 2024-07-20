import random

from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure
import pygad

number = int(input("what is the MNIST number you want?"))
data_num = 6000
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
def start_training(ga_instance, masklist, solution_idx):
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
        spectral_data_list.append(prediction_result)

    spectral_data_list = np.array(spectral_data_list)
    spectral_data_average = np.sum(spectral_data_list, axis=0) / len(spectral_data_list)
    spectral_data_squared_average = np.sum(spectral_data_list ** 2, axis=0) / len(spectral_data_list)
    spectral_data_std = np.sqrt(spectral_data_squared_average - spectral_data_average ** 2)
    result = sum(spectral_data_std)
    return result

desired_output = 0
num_generations = 30
num_parents_mating = 4

sol_per_pop = 8
num_genes = 10 #number of parameters

init_range_low = 0
init_range_high = 28*28-1

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10
'''
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=start_training,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print(start_training([],solution,[]))

for i in range(10):
    plt.imshow(spectral_data_list[i].reshape((64,64)))
    plt.show()
'''
guess = [412, 168, 673, 388, 230, 383, 698,  42, 465, 290, 424, 412,   1, 697,
 513, 221,  76, 385, 541, 157, 725, 571, 692, 545, 136, 123, 681, 201,
 608, 108, 309, 541, 769, 563, 621, 404,  97, 699, 134, 539, 637, 596,
 260, 605, 778, 419, 308, 645, 767, 487,  99, 778, 200, 369, 218, 619,
 577, 618,  91,  23, 661, 244, 735,  66, 338,  41, 126, 463, 668, 170,
 605, 261,  63, 246, 618, 225, 783, 206, 459, 255]
guess = [random.randint(0,28*28-1) for i in range(60)]
# guess = []
start_training([],guess,[])

count = 0
list1 = [[],[],[],[],[]]
for spectral_data in spectral_data_list:
    """
    if spectral_data[796] >= 0.01 and sum(spectral_data[921:925]) + sum(spectral_data[986:990]) > 0.1:
        count+=1
    """
    """
    a = sum(spectral_data[411:416])
    b = sum(spectral_data[601:605])
    c = sum(spectral_data[921:926])
    if (a+b > 0.02 or b+ c> 0.02) and sum(spectral_data[985:990]) > 0.04:
        count +=1
    """
    #6
    a = sum(spectral_data[408:417])
    b = sum(spectral_data[601:606])
    c = sum(spectral_data[794:799])
    d = sum(spectral_data[920:926])
    e = sum(spectral_data[985:993])
    if d+e > 0.02 and a > 0.03 and b+c+d > 0.02:
        count+=1
'''
for i in range(5):
    plt.hist(list1[i], bins=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
    plt.show()
'''
print(count)

for i in range(100):
    plt.plot(spectral_data_list[i], ".", alpha=0.3 )
plt.title(f"Mask for 1: number {number}")
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