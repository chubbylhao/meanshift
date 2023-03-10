import numpy as np


def euclidean_distance(x, y):
    """ 欧几里得距离不是尺度不变的，因此在使用此方式
    度量距离时要对数据进行归一化处理，另外，它在高维情
    况下变得不那么适用. """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    """ 它适用于高维数据间的距离度量. """
    return np.sum(np.abs((x - y)))


# TODO: 余弦相似度、汉明距离、切比雪夫距离
# TODO: ......


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.ones(4)
    res1 = euclidean_distance(a, b)
    res2 = manhattan_distance(a, b)
    print('', res1, '\n', res2)
