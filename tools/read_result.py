import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import scipy as sp

loaded_model = tf.keras.models.load_model("../tools/pretrained_model/model-BOHB-MAE.keras")
def show_mask(mask, num, con):
    pixels = np.zeros((28,28))
    for i in mask:
        i = int(i)
        pixels[i // 28][i % 28] = 255
    plt.imshow(pixels)
    plt.title(f"Mask for {num}vs{con}")
    plt.show()

def hyperspectra(mask, features, DRAW = 0):
    result = []
    for feature in features:
        for i in mask:
            i = int(i)
            feature[i // 28][i % 28] = 255
        prediction = loaded_model.predict(feature.reshape(1, 28, 28, 1), verbose=0).reshape(64, 64)
        prediction = prediction.flatten() / 100
        result.append(np.array(sp.special.softmax(prediction)))
    if DRAW:
        for i in range(min(len(result), 100)):
            plt.plot(result[i], ".", alpha=0.3)
        plt.show()
    return result

if __name__=="__main__":
    result_file = "../one_pick/PSO_result/2vs4_Sat.json"
    with open(result_file, "r") as f:
        mask = json.load(f)

    show_mask(mask["result"], 2, 4)

    import data_prep
    _, data1 = data_prep.get_MNIST(10, 2)
    hyperspectra(mask["result"], data1, DRAW=1)