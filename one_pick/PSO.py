'''
Optimisation using PSO (Particle Swarm Optimisation)
'''
import random
from sklearn import datasets
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
import skimage.measure
import time
import json
import pyswarms as ps

# if want debug ouputs, DEBUG = 1

# change this every time
number, contrast = 4, 6


class PSO():
    def __init__(self, DEBUG=1):
        self.DEBUG = DEBUG
        '''
        with open("../tools/peaks.json", "r") as f:
            self.peak = json.load(f)
            if self.DEBUG:
                print(f"peak count : {len(self.peak)}")
        '''
        self.peak = list(np.arange(1600,3000))
        self.loaded_model = tf.keras.models.load_model("../tools/pretrained_model/model-BOHB-MAE.keras")

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
                if i >= 28 * 28:
                    i -= 28 * 28
                i = int(i)
                number_feature[i // 28][i % 28] = 255
            prediction = self.loaded_model.predict(number_feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
            prediction = prediction / 100
            prediction = prediction.flatten()
            prediction = np.array(sp.special.softmax(prediction))[self.peak]
            for i, n in enumerate(prediction):
                if n > 0.01:
                    predictions[i] += 1
        '''
        plt.imshow(feature)
        plt.show()
        '''
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
            feature = np.array(feature).reshape(28, 28)
            for i in masklist:
                if i >= 28 * 28:
                    i -= 28 * 28
                i = int(i)
                feature[i // 28][i % 28] = 255
            prediction = self.loaded_model.predict(feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
            prediction = prediction.flatten() / 100
            prediction = np.array(sp.special.softmax(prediction))[self.peak]
            for i, n in enumerate(prediction):
                if n > 0.01:
                    predictions[i] -= 1
        return predictions

    def mask_format(self, mg):
        """
        binary mask to integer mask
        :param mg: binary mask
        :return: l: integer mask
        """
        l = []
        for n in range(len(mg)):
            if mg[n] == 1:
                l.append(n)
        return l

    def run(self, numbers, datas, swarm_size=15, options={'c1': 1.5, 'c2':1.5, 'w':0.5, 'k':4, 'p':1}, iters=10):
        '''
        Runs PSO
        :param numbers: [number, contrast], both should be int
        :param datas: [data_number, data_contrast], MNIST image data for both classes
        :param swarm_size: int, swarm size for PSO
        :param options: dict, options for PSO
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
        def start_training(masklist):
            masklist = self.mask_format(masklist)
            predictions = np.zeros(len(self.peak))

            predictions = self.count_peak_p(number_features, masklist, predictions)
            predictions = self.count_peak_m(contrast_features, masklist, predictions)
            if self.DEBUG:
                print(time.ctime())
            result = -max([-min(predictions), max(predictions)])
            return result


        def start_training_array(guess):
            '''
            PySwarm gives array of variables hence this function needs to make such loop
            :param guess: array, mask
            :return: array, loss functions
            '''
            l = []
            if len(guess) == 784:
                return start_training(guess)
            for g in guess:
                l.append(start_training(g))
            return np.array(l)

        guess_num = [random.randint(0,784) for i in range(100)]
        guess = []
        for i in range(784):
            if i in guess_num:
                guess.append(1)
            else:
                guess.append(0)

        if self.DEBUG:
            print(f"This is the first guess : {guess}")
            print(f"Loss of first guess: {start_training(guess)}")

        # PSO training
        dim = len(guess)

        optimizer = ps.discrete.binary.BinaryPSO(swarm_size, dim, options, init_pos=np.array([guess]))

        cost, binary_mask = optimizer.optimize(start_training_array, iters=iters)

        # Print end result
        print(f"Final Mask : {list(binary_mask)}")
        print(f"Loss of Final Mask : {cost}")
        print(f"Loss is better if closer to {-max(len(datas[0]),len(datas[1]))}")
        print(f"Total number of mask pixels : {sum(binary_mask)}")

        with open(f"PSO_result/{number}vs{contrast}_{time.ctime()}.json", "w") as f:
            json.dump({"train_size": len(number_features),"cost": cost, "result": self.mask_format(binary_mask)}, f)
        return self.mask_format(binary_mask)

    def test(self, numbers, datas, binary_mask, is_binary_mask = False):
        '''
        For testing the mask
        :param numbers: array[int,int]=[number, contrast]
        :param datas: array[data,data], MNIST data
        :param binary_mask: array, binary/integer mask array
        :param is_binary_mask: bool, default true,
        :return: int, loss function result
        '''
        if is_binary_mask:
            # transform binary mask into integer mask
            masklist = self.mask_format(binary_mask)
        else:
            # function input is integer mask, so just use it
            masklist = binary_mask

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
        result = -max([-min(predictions), max(predictions)])
        return [result, fp_fn]
