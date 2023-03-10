import cv2
import numpy as np
from meanshift import MeanShift
from convert_bgr2luv import bgr2luv
from convert_luv2bgr import luv2bgr
from utils import make_points_set, reconstruct_image
from kernel_functions import uniform_kernel, \
    normal_kernel, multivariate_normal_kernel
import matplotlib.pyplot as plt


def main():
    # 默认情形下，opencv读进来的灰度图是3通道的（因此省去了1通道转3通道的代码）
    img = cv2.imread('./image/raw/boat.png')
    img = cv2.resize(img, [170, 170], cv2.INTER_AREA)    # 受限于个人电脑的内存，这个值比较适合我

    # ------------------------------------------
    # 1. 先转换至Luv空间
    luv_image = bgr2luv(img)

    # 2. 由Luv格式的图像获取n×3格式的数组
    _, _, points_set = make_points_set(luv_image)

    # 3. 进行mean shift处理
    # 单核
    # mean_shift = MeanShift(points_set, kernel_function=normal_kernel)
    # new_points_set = mean_shift.fit(32)
    # 多元核
    mean_shift = MeanShift(points_set)
    new_points_set = mean_shift.fit(8, 16)

    # 4. 由处理好的n×3格式的数组获取处理后的Luv格式的图像
    new_luv_image = reconstruct_image(new_points_set)

    # 5. 转换回RGB空间
    res_image = luv2bgr(new_luv_image)

    # 6. 尝试进行分割
    gray_res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)
    # ------------------------------------------

    # ------------------------------------------
    # 并排对比处理前后的效果
    # res = np.hstack((img, res_image))
    # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 分别显示处理前后的效果
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img', 256, 256)
    cv2.namedWindow('res_image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('res_image', 256, 256)
    cv2.imshow('img', img)
    cv2.imshow('res_image', res_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ------------------------------------------


if __name__ == '__main__':
    main()
    # 下面是对分割的初步尝试
    # FIXME: 我暂时没有想到什么好的合并策略
    # img = cv2.imread('image/results/mandrill_results/(32, 16).jpg')
    # # img = cv2.imread('image/results/mandrill_results/original.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img, cmap='Accent')
    # plt.axis('off')
    # plt.show()
    # # cv2.imshow('img', img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
