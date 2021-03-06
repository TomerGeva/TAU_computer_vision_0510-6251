"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata

from ex1_functions import clip_and_place, transform_and_compute_distances_squared, clip_and_interp_backward

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        homogeneous = np.ones([1, np.size(match_p_src, axis=1)])
        # ==============================================================================================================
        # Creating matrix A
        # ==============================================================================================================
        src_points1     = np.transpose(np.insert(match_p_src, range(1, np.size(match_p_src, axis=1)+1), 0, axis=1))
        homogeneous1    = np.transpose(np.insert(homogeneous, range(1, np.size(match_p_src, axis=1)+1), 0, axis=1))
        src_points2     = np.transpose(np.insert(match_p_src, range(0, np.size(match_p_src, axis=1)), 0, axis=1))
        homogeneous2    = np.transpose(np.insert(homogeneous, range(0, np.size(match_p_src, axis=1)), 0, axis=1))
        mul_points1     = np.transpose(np.insert(match_p_src * match_p_dst[0, :], range(1, np.size(match_p_src, axis=1)+1), 0, axis=1))
        mul_points2     = np.transpose(np.insert(match_p_src * match_p_dst[1, :], range(0, np.size(match_p_src, axis=1)), 0, axis=1))
        i_points        = np.reshape(np.transpose(match_p_dst),(2*np.size(match_p_dst, axis=1), 1))

        a_mat = np.concatenate((-1*src_points1, -1*homogeneous1, -1*src_points2, -1*homogeneous2, mul_points1 + mul_points2, i_points), axis=1)

        # ==============================================================================================================
        # Performing SVD
        # ==============================================================================================================
        [_, _, v] = svd(a_mat)
        h = (v[-1].reshape((3, 3)))
        return h / h[-1, -1]

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        src_w, src_h, _ = src_image.shape
        points       = np.zeros((2, src_w * src_h))
        counter      = 0
        values       = np.matrix.reshape(src_image, (-1, 3), order='F')
        # ==============================================================================================================
        # Iterating over coordinates, performing the projection
        # ==============================================================================================================
        for ii in range(src_h):
            for jj in range(src_w):
                # --------------------------------------------------------------------------------------------------
                # Applying the transformation
                # --------------------------------------------------------------------------------------------------
                vec = np.array([[ii],[jj],[1]])
                dst = np.matmul(homography, vec)
                # --------------------------------------------------------------------------------------------------
                # Normalizing and placing in points
                # --------------------------------------------------------------------------------------------------
                points[:, counter] = [dst[0,0]/dst[2,0], dst[1,0]/dst[2,0]]
                counter += 1
        # ==============================================================================================================
        # Placing the points from the source in their new locations
        # ==============================================================================================================
        return clip_and_place(points, values, dst_image_shape)

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        src_h, src_w, _ = src_image.shape
        values = np.matrix.reshape(src_image, (-1, 3), order='F')
        yy, xx = np.meshgrid(np.arange(src_h), np.arange(src_w))
        input_flat = np.concatenate((xx.reshape((1, -1)), yy.reshape((1, -1)), np.ones_like(xx.reshape((1, -1)))), axis=0)
        # ==============================================================================================================
        # Computing transformation
        # ==============================================================================================================
        points = np.matmul(homography, input_flat)
        points_homogeneous = points[0:2, :] / points[2, :]
        del points
        # ==============================================================================================================
        # Interpolating for exact grid location ONLY FOR RELEVANT LOCATIONS
        # ==============================================================================================================
        return clip_and_place(points_homogeneous, values, dst_image_shape)

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # ==============================================================================================================
        # Local variable
        # ==============================================================================================================
        input_flat = np.concatenate((match_p_src, np.ones((1, match_p_src.shape[1]))), axis=0)
        # ==============================================================================================================
        # Computing  transformation and distances
        # ==============================================================================================================
        dist_squared = transform_and_compute_distances_squared(homography, input_flat, match_p_dst)
        # ==============================================================================================================
        # Computing metrics
        # ==============================================================================================================
        cond = dist_squared ** 0.5 < max_err

        fit_percent = np.sum(cond) / match_p_src.shape[1]
        if np.sum(cond) == 0:
            dist_mse = 10 ** 9 #if no points, return 10^9
        else:
            dist_mse = np.mean(dist_squared[cond])

        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # ==============================================================================================================
        # Local variable
        # ==============================================================================================================
        input_flat = np.concatenate((match_p_src, np.ones((1, match_p_src.shape[1]))), axis=0)
        # ==============================================================================================================
        # Computing  transformation and distances
        # ==============================================================================================================
        dist_squared = transform_and_compute_distances_squared(homography, input_flat, match_p_dst)
        # ==============================================================================================================
        # Isolating inliers
        # ==============================================================================================================
        cond = dist_squared ** 0.5 < max_err
        mp_src_meets_model = match_p_src[:, np.squeeze(cond)]
        mp_dst_meets_model = match_p_dst[:, np.squeeze(cond)]
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # ==============================================================================================================
        # Local variables, using class notations:
        # ==============================================================================================================
        w = inliers_percent
        t = max_err
        p = 0.99  # parameter determining the probability of the algorithm to succeed
        d = 0.5   # the minimal probability of points which meets with the model
        n = 4     # number of points sufficient to compute the model
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1 # number of RANSAC iterations (+1 to avoid the case where w=1)
        N = match_p_dst.shape[1]
        init_mse = 10**9
        src_best = dst_best = None
        np.random.seed(43)
        # ==============================================================================================================
        # Initial conditions
        # ==============================================================================================================
        best_homography =  self.compute_homography_naive(match_p_src, match_p_dst) #very initial homography - in case RANSAC parameters are bad
        best_mse        = init_mse
        best_fit        = d
        # ==============================================================================================================
        # Running k iterations
        # ==============================================================================================================
        for _ in range(k):
            # ------------------------------------------------------------------------------------------------------
            # Raffling 4 coordinate pairs
            # ------------------------------------------------------------------------------------------------------
            rand_indices = np.random.choice(N, n, False)
            # ------------------------------------------------------------------------------------------------------
            # Computing homography for this set
            # ------------------------------------------------------------------------------------------------------
            model = self.compute_homography_naive(match_p_src[:,rand_indices], match_p_dst[:,rand_indices])
            # cond = np.delete(np.arange(N),rand_indices)
            # ------------------------------------------------------------------------------------------------------
            # Testing the wellness of the model
            # ------------------------------------------------------------------------------------------------------
            # [fit_percent,dist_mse] = self.test_homography(model, match_p_src[:,cond], match_p_dst[:,cond],t)
            [fit_percent,dist_mse] = self.test_homography(model, match_p_src, match_p_dst,t)
            # ------------------------------------------------------------------------------------------------------
            # If the model is the best so far, updates
            # ------------------------------------------------------------------------------------------------------
            if (fit_percent > best_fit) or (fit_percent == best_fit and dist_mse < best_mse):
                src_best, dst_best  = self.meet_the_model_points(model, match_p_src, match_p_dst, t)
                best_mse            = dist_mse
                best_fit            = fit_percent
                best_homography     = model
        # ==============================================================================================================
        # Creating a homography with all the best inliers
        # ==============================================================================================================
        if src_best is not None:
            model_temp = self.compute_homography_naive(src_best, dst_best)
            [fit_percent, dist_mse] = self.test_homography(model_temp, match_p_src, match_p_dst, t)
            if (fit_percent > best_fit) or (fit_percent == best_fit and dist_mse < best_mse):
                return model_temp
        return best_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        src_w, src_h, _ = src_image.shape
        yy, xx = np.meshgrid(np.arange(dst_image_shape[0]), np.arange(dst_image_shape[1]))
        dst_meshgrid_flat = np.concatenate((xx.reshape((1, -1)), yy.reshape((1, -1)), np.ones_like(xx.reshape((1, -1)))), axis=0)
        # ==============================================================================================================
        # Computing transformation
        # ==============================================================================================================
        points = np.matmul(backward_projective_homography, dst_meshgrid_flat)
        points_homogeneous = points[0:2, :] / points[2, :]
        del points
        # ==============================================================================================================
        # Clip relevant points and interpolate for exact grid location ONLY FOR RELEVANT LOCATIONS
        # ==============================================================================================================
        backward_warp =  clip_and_interp_backward(points_homogeneous, src_image, dst_meshgrid_flat, dst_image_shape)

        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        pad_up = pad_down = pad_right = pad_left = 0
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {'upper_left':  np.array([[0], [0], [1]]),
                     'upper_right': np.array([[src_cols_num - 1], [0], [1]]),
                     'lower_left':  np.array([[0], [src_rows_num - 1], [1]]),
                     'lower_right': np.array([[src_cols_num - 1], [src_rows_num - 1], [1]])}
        transformed_edges = {}
        # ==============================================================================================================
        # Transforming the edges to the destination coordinates
        # ==============================================================================================================
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        # ==============================================================================================================
        # Extracting the needed padding
        # ==============================================================================================================
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 0:
                # --------------------------------------------------------------------------------------------------
                # pad up
                # --------------------------------------------------------------------------------------------------
                pad_up = max([pad_up, 1 + abs(corner_location[1])])
            if corner_location[0] >= dst_cols_num:
                # --------------------------------------------------------------------------------------------------
                # pad right
                # --------------------------------------------------------------------------------------------------
                pad_right = max([pad_right,1 +  corner_location[0] - dst_cols_num])
            if corner_location[0] < 0:
                # --------------------------------------------------------------------------------------------------
                # pad left
                # --------------------------------------------------------------------------------------------------
                pad_left = max([pad_left, 1 + abs(corner_location[0])])
            if corner_location[1] >= dst_rows_num:
                # --------------------------------------------------------------------------------------------------
                # pad down
                # --------------------------------------------------------------------------------------------------
                pad_down = max([pad_down, 1 + corner_location[1] - dst_rows_num])
        # ==============================================================================================================
        # Computing panorama size
        # ==============================================================================================================
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        translation = np.concatenate((np.eye(3,2),np.array([[-1 * pad_left], [-1 * pad_up], [1]])), axis=1)
        # translation = 0
        final_homography = np.matmul(backward_homography, translation)
        final_homography /=  final_homography[2,2]
        # final_homography = backward_homography + translation
        return final_homography


    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # ==============================================================================================================
        # Find the RANSAC Homography
        # ==============================================================================================================
        ransac_homography = self.compute_homography(match_p_src,
                                                    match_p_dst,
                                                    inliers_percent,
                                                    max_err)
        # ==============================================================================================================
        # Find the Panorama shape and needed padding
        # ==============================================================================================================
        [panorama_h, panorama_w, pad_struct] = self.find_panorama_shape(src_image,
                                                                        dst_image,
                                                                        ransac_homography)
        # ==============================================================================================================
        # Find the Proper Backward Homography with needed translation
        # ==============================================================================================================
        backward_homography = self.add_translation_to_backward_homography(np.linalg.inv(ransac_homography),
                                                                          pad_struct.pad_left,
                                                                          pad_struct.pad_up)
        # ==============================================================================================================
        # Execute Backward Warp to create [panorama_h X panorama_w] image with the src
        # ==============================================================================================================
        src_image_to_panorama_backward_warp = self.compute_backward_mapping(backward_homography,
                                                                            src_image,
                                                                            (panorama_h, panorama_w, 3))
        # ==============================================================================================================
        # Adding the destination pixels on top of the transformed source
        # ==============================================================================================================
        img_panorama = src_image_to_panorama_backward_warp
        xx, yy = np.meshgrid(np.arange(dst_image.shape[1]), np.arange(dst_image.shape[0]))
        yy += pad_struct.pad_up
        xx += pad_struct.pad_left
        img_panorama[yy, xx, :] = dst_image

        return img_panorama
