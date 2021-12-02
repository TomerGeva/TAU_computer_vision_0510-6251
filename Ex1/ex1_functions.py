import numpy as np
from scipy.interpolate import griddata


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
    cond = np.all(points > -1, axis=0)
    values_relevant = values[cond, :]
    points_relevant = points[:, cond]
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