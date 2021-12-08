import numpy as np
from scipy.interpolate import griddata


def clip_relevant(points: np.ndarray,
                  values: np.ndarray,
                  dst_image_shape: tuple,
                  gt_half: bool = False) -> tuple:
    """
    :param points: 2 X (src_w * src_h) array where the ith column represents a transformed coordinates
    :param values: (src_w * src_h) X 3 array where the ith row represents a RGB pixel of the source picture
    :param dst_image_shape: Shape of the wanted picture after interpolation
    :param gt_half: if true sets the threshold on -0.5 and not -1
    :return: only relevant points and matching values
    """
    threshold = -0.5 if gt_half else -1
    cond = np.all(np.concatenate((points > threshold,
                                  np.expand_dims(points[0,:] < (dst_image_shape[1] - (1 + threshold)), axis=0),
                                  np.expand_dims(points[1,:] < (dst_image_shape[0] - (1 + threshold)), axis=0)),
                                 axis=0),
                  axis=0)
    values_relevant = values[cond, :]
    points_relevant = points[:, cond]
    return points_relevant, values_relevant

def clip_and_interp(points: np.ndarray,
                    values: np.ndarray,
                    dst_image_shape: tuple) -> np.ndarray:
    """
    :param points: 2 X (src_w * src_h) array where the ith column represents a transformed coordinates
    :param values: (src_w * src_h) X 3 array where the ith row represents a RGB pixel of the source picture
    :param dst_image_shape: Shape of the wanted picture after interpolation
    :return: The destination photo post interpolation
    """
    # ==============================================================================================================
    # Local variables
    # ==============================================================================================================
    yy, xx = np.meshgrid(np.arange(dst_image_shape[1]), np.arange(dst_image_shape[0]))
    # ==============================================================================================================
    # Clipping only relevant point for speedup
    # ==============================================================================================================
    points_relevant, values_relevant = clip_relevant(points, values, dst_image_shape)
    # ==============================================================================================================
    # Interpolating
    # ==============================================================================================================
    src_image_warp = griddata(np.transpose(points_relevant), values_relevant, (yy, xx), method='linear')
    # ==============================================================================================================
    # Numerical rounding etc
    # ==============================================================================================================
    src_image_warp[np.isnan(src_image_warp)] = 0
    src_image_warp[src_image_warp > 255] = 255
    src_image_warp = np.uint8(src_image_warp)
    return src_image_warp


def clip_and_place(points: np.ndarray,
                   values: np.ndarray,
                   dst_image_shape: tuple) -> np.ndarray:
    """
    :param points: 2 X (src_w * src_h) array where the ith column represents a transformed coordinates
    :param values: (src_w * src_h) X 3 array where the ith row represents a RGB pixel of the source picture
    :param dst_image_shape: Shape of the wanted picture after interpolation
    :return: ndarray with the size of the destination picture, where all the transformed points are placed
    """
    # ==============================================================================================================
    # Local variables
    # ==============================================================================================================
    dest_img = np.zeros(dst_image_shape).astype(int)
    # ==============================================================================================================
    # Clipping only relevant point for speedup
    # ==============================================================================================================
    points_relevant, values_relevant = clip_relevant(points, values, dst_image_shape, gt_half=True)
    # ==============================================================================================================
    # Rounding and placing
    # ==============================================================================================================
    points_relevant = (np.round(points_relevant)).astype(int)
    dest_img[points_relevant[1,:], points_relevant[0,:], :] = np.uint8(values_relevant)
    return dest_img

def clip_and_interp_backward(points: np.ndarray,
                             src_image: np.ndarray,
                             flat_meshgrid_dst: np.ndarray,
                             dst_image_shape: tuple) -> np.ndarray:
    """
    :param points: 2 X (dst_w * dst_h) array where the ith column represents a transformed coordinates
    :param src_image: source image to determine the RGB value of the relevant points
    :param flat_meshgrid_dst:
    :param dst_image_shape: Shape of the wanted picture after interpolation
    :return: ndarray with the size of the destination picture, where all the transformed points are placed
    """
    # ==============================================================================================================
    # Local variables
    # ==============================================================================================================
    dest_img = np.zeros(dst_image_shape).astype(int)
    threshold = -1
    # ==============================================================================================================
    # Clipping only relevant points for speedup
    # ==============================================================================================================
    cond = np.all(np.concatenate((points >= threshold,
                                  np.expand_dims(points[0, :] < (src_image.shape[1] - (1 + threshold)), axis=0),
                                  np.expand_dims(points[1, :] < (src_image.shape[0] - (1 + threshold)), axis=0)),
                                 axis=0),
                  axis=0)
    points_relevant = points[:, cond]
    xx, yy = np.meshgrid(np.arange(src_image.shape[1]), np.arange(src_image.shape[0])) #Source img coordinate meshgrid
    input_flat = np.concatenate((xx.reshape((1, -1)), yy.reshape((1, -1))))
    src_values_in_src_coor = np.matrix.reshape(src_image, (-1, 3), order='C')
    # ==============================================================================================================
    # Interpolating
    # ==============================================================================================================
    values_relevant = griddata(np.transpose(input_flat), src_values_in_src_coor, np.transpose(points_relevant), method='cubic', fill_value=0)
    # values_relevant[np.isnan(values_relevant)] = 0
    values_relevant[values_relevant > 255] = 255
    points_relevant_dst = flat_meshgrid_dst[:, cond]
    # ==============================================================================================================
    # Rounding and placing
    # ==============================================================================================================
    dest_img[points_relevant_dst[1,:], points_relevant_dst[0,:], :] = np.uint8(np.round(values_relevant))
    return dest_img


def transform_and_compute_distances_squared(homography: np.ndarray,
                                            source: np.ndarray,
                                            destination: np.ndarray) -> np.ndarray:
    """
    :param homography: 3x3 Projective Homography matrix.
    :param source: 3 X N matrix where each column is a set of homogeneous coordinates
    :param destination: 2 X N matrix where each column matches the respective source column coordinate location after the transformation
    :return: the function returns a vector of the squared distances of the transformed points and the destination points
    """
    # ==============================================================================================================
    # Computing transformation
    # ==============================================================================================================
    points = np.matmul(homography, source)
    points = points[0:2, :] / points[2, :]
    # ==============================================================================================================
    # Computing distances
    # ==============================================================================================================
    return (points[0, :] - destination[0, :]) ** 2 + (points[1, :] - destination[1:]) ** 2