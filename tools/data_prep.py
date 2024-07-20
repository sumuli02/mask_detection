'''
Prepareing MNIST data
'''
import numpy as np
from sklearn import datasets
import cv2

data = datasets.fetch_openml('mnist_784', version=1, as_frame=False,parser='auto', return_X_y=True)

def get_MNIST(total_num, filter=""):
    """
    Get MNIST label and MNIST data
    :param total_num: int, total number of MNIST data you want
    :param filter: int, class of MNIST data you want, default is all class
    :return: MNIST label and MNIST data
    """
    if filter != "":
        data_num = int(total_num * 12)
    else:
        data_num = total_num
    m_data = np.array(data[0][:data_num])
    m_label = np.array(data[1][:data_num])
    if filter != "":
        filter = np.where(m_label == str(filter))
        m_label, m_data = m_label[filter], np.array(m_data)[filter]

    m_data = [cv2.threshold(i.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].reshape(28, 28,1) for i in m_data]
    m_label = [int(i) for i in m_label[1]]
    return m_label[:total_num], m_data[:total_num]