import numpy as np


def luv2bgr(luv_image):
    """将Luv格式转换为BGR格式.

    Args:
        luv_image: 经mean shift处理后得到的Luv格式的图像.

    Returns:
        BGR格式的图像.
    """
    # 1. Luv2XYZ
    L, u, v = np.split(luv_image, 3, 2)
    L = 100 / 255 * L
    u = 354 / 255 * u - 134
    v = 262 / 255 * v - 140
    # 保持数值稳定性的措施
    epsilon = 1e-9
    L = np.where(L == 0, epsilon, L)
    U = u / (13 * L) + 0.19793943
    V = v / (13 * L) + 0.46831096
    Y = np.where(L > 8, ((L + 16) / 116) ** 3, 0.00110706 * L)
    X = Y * (9 * U) / (4 * V)
    Z = Y * (12 - 3 * U - 20 * V) / (4 * V)

    # 2. XYZ2BGR
    r = +3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0411560 * Z
    b = +0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    image = np.concatenate((b, g, r), 2)
    image = (image - image.min()) / \
            (image.max() - image.min())

    # 显示统一使用uint8格式，而不使用在[0,1]之间的浮点格式
    # FIXME: 转换成uint8格式
    return (image * 255).astype(np.uint8)


if __name__ == '__main__':
    import cv2
    from convert_bgr2luv import bgr2luv
    img = cv2.imread('./image/raw/mandrill.png')
    luv_img = bgr2luv(img)
    res_image = luv2bgr(luv_img)
    cv2.imshow('luv_image', luv_img)
    cv2.imshow('res_image', res_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
