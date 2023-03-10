import numpy as np
from kernel_functions import uniform_kernel, \
    normal_kernel, multivariate_normal_kernel


class MeanShift:
    def __init__(
            self,
            points_set,
            kernel_function=multivariate_normal_kernel,
            max_iterations=3,
            mode='image_processing',
    ):
        self.points_set = points_set
        self.kernel_function = kernel_function
        self.max_iterations = max_iterations
        if mode == 'image_processing':
            self.scale_factor = np.ceil((points_set[:, :2].max() / 255))
            self.points_set[:, :2] = points_set[:, :2] / self.scale_factor
        elif mode == 'points_processing':
            # TODO: 我关注的是mean shift在图像中的应用，而在一般点集
            # TODO: 中的应用则需要根据个人手头上的数据的实际特点做一些相应的变化
            pass
        else:
            raise Exception("Mode that don't exist!")

    def weights_matrix(self):
        """

        每次迭代不可能遍历所有的点（那样算法执行起来太低效），因此，此函数
        的作用是基于像素点的空间xy坐标计算出一个小的贪心邻域（比如400个邻域点），
        这样做是合理的，因为空间上相邻的像素点同属一类的概率比较大（不过这样的
        假设也让算法偏心于空间坐标之间的联系）.

        Returns:
            贪心邻域点的索引.
        """
        rows, _ = self.points_set.shape
        matrix = np.zeros((rows, rows))
        for k in range(2):
            col_1dim = self.points_set[:, k]
            col_2dims = np.expand_dims(col_1dim, 1)
            matrix += (col_2dims - col_1dim) ** 2
        # 贪心邻域内的点数（比如取400个点）
        # 相当于将范围限定在一个20×20的小图像块内
        points_num = 20 * 20
        indices = np.argsort(matrix, 1)[:, 1:points_num + 1]

        return indices

    # TODO: 单核分割效果不太好
    def one_bandwidth(self, h):
        pass

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FIXME: 冗余代码，需要重构（解决不定参数个数问题）
    def two_bandwidth(self, hs, hr):
        indices = self.weights_matrix()
        new_points_set = np.zeros(self.points_set.shape)
        for k in range(self.points_set.shape[0]):
            ci = self.points_set[k]
            yij = self.points_set[indices[k]]
            flag = 1
            while flag <= self.max_iterations:
                weights = self.kernel_function(ci, yij, hs, hr)
                next_ci = np.sum(weights * yij, axis=0) / np.sum(weights)
                # 更新kernel window的中心点
                ci = next_ci
                flag += 1
                # TODO: 迭代的阈值停止条件（这里决定迭代次数就好了），至于阈值，想加的可以加上
            new_points_set[k] = ci

        return new_points_set

    def fit(self, *args):
        if len(args) == 1:
            h = args[0]
            new_points_set = self.one_bandwidth(h)
        elif len(args) == 2:
            hs, hr = args
            new_points_set = self.two_bandwidth(hs, hr)
        else:
            raise Exception('The number of parameters is incorrect.')

        new_points_set[:, :2] = np.round(self.points_set[:, :2] * self.scale_factor)

        return new_points_set


if __name__ == '__main__':
    pass
