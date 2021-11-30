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
    yy, xx = np.meshgrid(np.arange(dst_image_shape[0]), np.arange(dst_image_shape[1]))
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