import numpy as np


def uniform_kernel(ci, yij, h):
    """按均匀分布核函数计算用于更新下一个迭代点的其它点的权重.

    假设当前迭代点为ci，除当前迭代点以外的其它点为yij，则该函
    数为每一个yij计算一个权重，这些权重用于加权更新下一个迭代点.

    Args:
        ci: ndarray，当前迭代点（kernel window的中心点）.
        yij: ndarray，除当前迭代点以外的其它点.
        h: 核函数的bandwidth

    Returns:
        对应于每一个yij的权重.
    """
    distance = np.sum(((ci - yij) / h) ** 2, axis=1, keepdims=True)
    return np.ones((yij.shape[0], 1)) * (distance <= 1).astype(int)


def normal_kernel(ci, yij, h):
    """按一元高斯分布核函数计算.

    因为是做加权平均，高斯函数前面的标准化常数可以约去，故
    在以下的定义中，默认此标准化常数的值为1.

    Args:
        ci:
        yij:
        h:

    Returns:

    """
    distance = np.sum(((ci - yij) / h) ** 2, axis=1, keepdims=True)
    return np.exp(-0.5 * distance)


def multivariate_normal_kernel(ci, yij, hs, hr):
    """按多元高斯分布核函数计算.

    Args:
        ci:
        yij:
        hs: 坐标（一般情况下只有x和y两个坐标）的bandwidth.
        hr: 颜色空间（一般情况下是Luv颜色空间）的bandwidth.

    Returns:

    """
    distance_hs, distance_hr = \
        np.sum(((ci[:2] - yij[:, :2]) / hs) ** 2, axis=1, keepdims=True), \
        np.sum(((ci[2:] - yij[:, 2:]) / hr) ** 2, axis=1, keepdims=True)
    return np.exp(-0.5 * (distance_hs + distance_hr))


if __name__ == '__main__':
    a = np.array([0, 0, 0])
    b = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    res1 = uniform_kernel(a, b, 1)
    res2 = normal_kernel(a, b, 1)
    res3 = multivariate_normal_kernel(a, b, 1, 2)
    print('', res1, '\n\n', res2, '\n\n', res3)
