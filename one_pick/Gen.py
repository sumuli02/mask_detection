'''
One pick
Optimisation using genetic algorithm
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
import json

class GenAlg():
    def __init__(self, DEBUG=1):
        self.DEBUG = DEBUG
        self.peak = list(np.arange(1600, 3000))
        self.loaded_model = tf.keras.models.load_model("../tools/pretrained_model/model-BOHB-MAE.keras")

    def mask_gen(self, mg):
        m =[]
        for i in mg:
            i *= 200
            i = int(i)
            while i >= 28 * 28:
                i -= 28 * 28
            while i < 0:
                i += 28 * 28
            m.append(i)
        return m

    def count_peak_p(self, features, masklist, predictions):
        '''
        count peak by +1
        :param features:
        :param masklist:
        :param predictions:
        :return:
        '''
        for feature in features:
            number_feature = np.array(feature).reshape(28, 28)
            for i in masklist:
                number_feature[i // 28][i % 28] = 255
            prediction = self.loaded_model.predict(number_feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
            prediction = prediction.flatten() / 100
            prediction = np.array(sp.special.softmax(prediction))[self.peak]
            for i, n in enumerate(prediction):
                if n > 0.01:
                    predictions[i] += 1
        return predictions

    def count_peak_m(self, features, masklist, predictions):
        '''
        count peak by -1
        :param features:
        :param masklist:
        :param predictions:
        :return:
        '''
        for feature in features:
            number_feature = np.array(feature).reshape(28, 28)
            for i in masklist:
                number_feature[i // 28][i % 28] = 255
            prediction = self.loaded_model.predict(number_feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
            prediction = prediction.flatten() / 100
            prediction = np.array(sp.special.softmax(prediction))[self.peak]
            for i, n in enumerate(prediction):
                if n > 0.01:
                    predictions[i] -= 1
        return predictions

    def run(self, numbers, datas, pixel_num=50,  num_generations=15, num_parents_mating=3, sol_per_pop=5):
        '''
        Runs GenAlg
        :param numbers: [number, contrast], both should be int
        :param datas: [data_number, data_contrast], MNIST image data for both classes
        :return:
        '''
        number, contrast = numbers[0], numbers[1]
        number_features, contrast_features = datas[0], datas[1]

        if self.DEBUG:
            print(f"Two Classes Compared: {number} and {contrast}")
            print(f"Training set: \n  "
                  f"Class {number} : {len(number_features)}\n  "
                  f"Class {contrast} : {len(contrast_features)}")

        ## training part
        def start_training(ga_instance, masklist, solution_idx):
            predictions = np.zeros(len(self.peak))
            masklist = self.mask_gen(masklist)

            predictions = self.count_peak_p(number_features, masklist, predictions)
            predictions = self.count_peak_m(contrast_features, masklist, predictions)

            if self.DEBUG:
                print(time.ctime())
            result = max([-min(predictions), max(predictions)])
            return result

        # GenAlg training
        num_genes = pixel_num
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               fitness_func=start_training,
                               parent_selection_type='sss',
                               crossover_type='single_point',
                               keep_parents=1)

        ga_instance.run()

        ga_instance.plot_fitness(label=['Fitting'])

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        # Print end result
        print(f"Final Mask : {list(solution)}")
        print(f"Loss of Final Mask : {solution_fitness}")
        print(f"Loss is better if closer to {max(len(datas[0]), len(datas[1]))}")
        print(f"Total number of mask pixels : {pixel_num}")

        with open(f"Gen_result/{number}vs{contrast}_{time.ctime()}.json", "w") as f:
            json.dump({"train_size": len(number_features), "cost": solution_fitness, "result": self.mask_gen(solution)}, f)
        return self.mask_gen(solution)

    def test(self, numbers, datas, masklist):
        '''
        For testing the mask
        :param numbers: array[int,int]=[number, contrast]
        :param datas: array[data,data], MNIST data
        :param binary_mask: array, binary/integer mask array
        :param is_binary_mask: bool, default true,
        :return: int, loss function result
        '''
        predictions = np.zeros(len(self.peak))

        predictions = self.count_peak_p(datas[0], masklist,predictions)
        pre = predictions.copy()
        print(pre)
        predictions = self.count_peak_m(datas[1], masklist,predictions)
        print(predictions)
        aft = -predictions + pre
        maxi = max(predictions)
        mini = -min(predictions)
        if (maxi < mini):
            index = np.argmin(predictions)
        else:
            index = np.argmax(predictions)

        print(f'Test Result (count, total) {numbers[0]}vs{numbers[1]}')
        print(f'({pre[index]} , {len(datas[0])}) vs ({aft[index]} , {len(datas[1])})')
        print(f'{index} index')
        r1 = pre[index]/len(datas[0])
        r2 = aft[index]/len(datas[1])
        # false positive + false negative
        fp_fn = 1-max([r1,r2])+min([r1,r2])
        print(f"False positive:{1-max([r1,r2])}")
        print(f"False negative:{min([r1,r2])}\n\n")
        if self.DEBUG:
            print(time.ctime())
        result = max([-min(predictions), max(predictions)])
        return [result, fp_fn]
