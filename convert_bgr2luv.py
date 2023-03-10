import numpy as np


def bgr2luv(image):
    """BGR转Luv.

    首先需要将BGR空间转化为XYZ空间，再将XYZ空间转化为Luv空间.

    Args:
        image: int类型、bgr格式的图像.

    Returns:
        Luv格式的图像.
    """
    # 1. BGR2XYZ
    # 整幅图像缩放至[0,1]区间
    image = (image - image.min()) / \
            (image.max() - image.min())
    b, g, r = np.split(image, 3, 2)
    # 每个通道分别缩放至[0,1]区间
    # b = (b - b.min()) / (b.max() - b.min())
    # g = (g - g.min()) / (g.max() - g.min())
    # r = (r - r.min()) / (r.max() - r.min())
    # 为了避免0作分母所采取的数值稳定性措施
    epsilon = 1e-9
    zero_points = (b == 0) & (g == 0) & (r == 0)
    b = np.where(zero_points, np.zeros(b.shape) + epsilon, b)
    g = np.where(zero_points, np.zeros(g.shape) + epsilon, g)
    r = np.where(zero_points, np.zeros(r.shape) + epsilon, r)
    # XYZ（不可能出现X=Y=Z=0的情况）
    # 此实现参考了OpenCV的官方文档：https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    # 其中RGB Working Space为sRGB，Reference White为D65
    # 详情参考：http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # 维基百科：https://en.wikipedia.org/wiki/CIELUV
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    # 2. XYZ2Luv
    L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, 903.3 * Y)
    U = (4 * X) / (X + 15 * Y + 3 * Z)
    V = (9 * Y) / (X + 15 * Y + 3 * Z)
    u = 13 * L * (U - 0.19793943)
    v = 13 * L * (V - 0.46831096)
    # 转换为8bit数据类型
    L = 255 / 100 * L
    u = 255 / 354 * (u + 134)
    v = 255 / 262 * (v + 140)

    # 3. 返回Luv格式的图像，各值在[0,255]之间
    # 不可避免地会产生一点舍入误差（有更好的方法请告诉我）
    return np.concatenate((L, u, v), 2).astype(np.uint8)


if __name__ == '__main__':
    import cv2
    img = cv2.imread('./image/raw/mandrill.png')
    luv_img = bgr2luv(img)
    luv_img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    # 对比自己的bgr2luv实现与OpenCV官方的COLOR_BGR2Luv实现
    res = np.hstack((img, luv_img, luv_img_cv2))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
