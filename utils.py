import numpy as np


def make_points_set(luv_image):
    """将Luv空间的图像转换为易于操作的特征空间中的点的集合.

    很显然，luv_image的形状为(Height×Width×Channels)，不利于处理，而
    若将其转换到一个5维特征空间，即(x,y,L,u,v)，则可以方便地进行聚类.

    Args:
        luv_image: 由convert_bgr2luv函数返回的图像.

    Returns:
        一个形状如(n×5)的ndarray数组，其中n是图像中的像素数量.
    """
    luv_image = np.transpose(luv_image, (2, 1, 0))
    luv_points_set = np.reshape(luv_image.T, (-1, 3))
    rows, cols = luv_image.shape[1], luv_image.shape[2]
    x_coordinates = np.array(range(1, rows + 1))
    x_coordinates = np.expand_dims(x_coordinates, axis=1)
    x_coordinates = np.tile(x_coordinates, (cols, 1))
    y_coordinates = np.array(range(1, cols + 1))
    y_coordinates = np.expand_dims(y_coordinates, axis=1)
    y_coordinates = np.repeat(y_coordinates, rows, axis=0)
    xy_points_set = np.concatenate((x_coordinates, y_coordinates), axis=1)
    points_set = np.concatenate((xy_points_set, luv_points_set), axis=1)

    return xy_points_set, luv_points_set, points_set


def reconstruct_image(points_set):
    """从点集恢复图像.

    Args:
        points_set: 经mean shift算法处理完之后的点集.

    Returns:
        Luv格式的图像.
    """
    xy_points_set, luv_points_set = points_set[:, :2], points_set[:, 2:]
    luv_image = np.reshape(luv_points_set, (xy_points_set[-1, -1].astype(int), -1, 3)).T
    luv_image = np.transpose(luv_image, (2, 1, 0))

    return luv_image.astype(np.uint8)


if __name__ == '__main__':
    import cv2
    from convert_bgr2luv import bgr2luv
    img = cv2.imread('./image/raw/mandrill.png')
    luv_img = bgr2luv(img)
    xy, luv, xy_luv = make_points_set(luv_img)
    res_luv_img = reconstruct_image(xy_luv)
    cv2.imshow('res_luv_img', res_luv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
