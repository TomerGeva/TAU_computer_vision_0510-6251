"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d, medfilt2d

global WIN_SIZE


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int,
                     get_median: bool = False) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).

            right = np.expand_dims(np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]), axis=2)
            left = np.expand_dims(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]), axis=2)
            ssdd = solution.ssd_distance(left.astype(np.float64), right.astype(np.float64), win_size=WIN_SIZE, dsp_range=2)
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        kernel = np.ones((win_size, win_size))
        # ==============================================================================================================
        # Padding the right image along the x axis
        # ==============================================================================================================
        right_image_pad = np.pad(right_image, ((0, 0), (dsp_range, dsp_range), (0, 0)))
        # ==============================================================================================================
        # Iterating over the disparity vector, computing SSD
        # ==============================================================================================================
        for ii, disparity in enumerate(disparity_values):
            # ------------------------------------------------------------------------------------------------------
            # Preparing the images for the convolution
            # ------------------------------------------------------------------------------------------------------
            images_diff_squared = np.sum(np.power(left_image - right_image_pad[:, ii:ii + right_image.shape[1]], 2), axis=2)
            # ------------------------------------------------------------------------------------------------------
            # Performing the convolution one dimension at a time since we can not use pytorch :/
            # ------------------------------------------------------------------------------------------------------
            if not get_median:
                ssdd_tensor[:, :, ii] = convolve2d(images_diff_squared, kernel, mode='same')
            else:
                ssdd_tensor[:, :, ii] = medfilt2d(images_diff_squared, kernel_size=win_size)
                # ==============================================================================================================
        # Normalizing to range
        # ==============================================================================================================
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        return np.argmin(ssdd_tensor, axis=2)

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        try:
            num_labels, num_of_cols = c_slice.shape[1], c_slice.shape[0]
        except IndexError:
            num_labels = c_slice.shape[0]
            num_of_cols = 1
            c_slice = np.expand_dims(c_slice, axis=0)
        l_slice = np.zeros((num_labels, num_of_cols))
        yy, xx = np.meshgrid(np.arange(num_labels), np.arange(num_labels))
        # ==============================================================================================================
        # Filling the loss matrix in a for loop since each column requires the previous column
        # ==============================================================================================================
        for col, c_clise_col in enumerate(c_slice):
            if col == 0:
                l_slice[:, col] = c_clise_col
            else:
                # --------------------------------------------------------------------------------------------------
                # Computing the transition matrix
                # --------------------------------------------------------------------------------------------------
                # **********************************************************************************************
                # Filling initial values without penalties
                # **********************************************************************************************
                transition_matrix = np.tile(l_slice[:, col-1], [num_labels, 1])
                # **********************************************************************************************
                # Adding P1 for deviation from the main diagonal by +-1
                # **********************************************************************************************
                transition_matrix[np.abs(yy - xx) == 1] += p1
                # **********************************************************************************************
                # Adding P2 for deviation from the main diagonal by +-2 or more
                # **********************************************************************************************
                transition_matrix[np.abs(yy - xx) >= 2] += p2
                # --------------------------------------------------------------------------------------------------
                # Inserting the minimum value matching each label in the mlse matrix
                # --------------------------------------------------------------------------------------------------
                #          C_{slice}(d, col)  +         M(d,col)                  - min(L(:,col-1))
                l_slice[:, col] = c_clise_col + np.min(transition_matrix, axis=1) - np.min(l_slice[:, col - 1])

        return l_slice

    @staticmethod
    def extract_slices(ssdd_tensor: np.ndarray,
                       direction: int,
                       transpose: bool = False,
                       fliplr: bool = False,
                       flipud: bool = False):
        """
        :param ssdd_tensor: 3D SSDD tensor with shape (H, W, label_size)
        :param direction: a number between 1 and 2:
            1. left     -> right
            2. top left -> bottom right
        :param transpose: if true, transposes the indices in the dictionary
        :param fliplr: if true, flips the indices in the dictionary left -> right
        :param flipud: if true, flips the indices in the dictionary top  -> bottom
        :return:
            1. A dictionary with number keys where each key matches a slice
            2. A dictionary with number keys where each key matches a slice's indices in the photo
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        height, width, labels_num = ssdd_tensor.shape
        slices_dict = {}
        indices_dict = {}
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # ==============================================================================================================
        # Performing according to each direction
        # ==============================================================================================================
        if direction == 1:
            for ii in range(height):
                slices_dict[ii] = ssdd_tensor[ii, :, :]
                indices_dict[ii] = np.array(list(zip(yy[ii, :], xx[ii, :])))
                # --------------------------------------------------------------------------------------------------
                # Dealing with flipped input left -> right
                # --------------------------------------------------------------------------------------------------
                if fliplr:
                    indices_dict[ii][:, 1] = width - 1 - indices_dict[ii][:, 1]
                # --------------------------------------------------------------------------------------------------
                # Dealing with flipped input top -> bottom
                # --------------------------------------------------------------------------------------------------
                if flipud:
                    indices_dict[ii][:, 0] = height - 1 - indices_dict[ii][:, 0]
                # --------------------------------------------------------------------------------------------------
                # Dealing with transposed input
                # --------------------------------------------------------------------------------------------------
                if transpose:
                    indices_dict[ii] = np.fliplr(indices_dict[ii])

        elif direction == 2:
            counter = 0
            for ii in range(np.max([height, width])):
                indices_temp = np.array(list(zip(range(np.max([height, width])), range(np.max([height, width])))))[ii:,
                               :]
                indices_temp[:, 1] -= ii
                indices_dict[counter] = indices_temp[indices_temp[:, 0] < height, :] if height <= width else indices_temp[indices_temp[:, 1] < width, :]
                slices_dict[counter] = ssdd_tensor[indices_dict[counter][:, 0], indices_dict[counter][:, 1], :]
                # --------------------------------------------------------------------------------------------------
                # Dealing with flipped input left -> right
                # --------------------------------------------------------------------------------------------------
                if fliplr:
                    indices_dict[counter][:, 1] = width - 1 - indices_dict[counter][:, 1]
                # --------------------------------------------------------------------------------------------------
                # Dealing with flipped input top -> bottom
                # --------------------------------------------------------------------------------------------------
                if flipud:
                    indices_dict[counter][:, 0] = height - 1 - indices_dict[counter][:, 0]
                # --------------------------------------------------------------------------------------------------
                # Dealing with transposed input
                # --------------------------------------------------------------------------------------------------
                if transpose:
                    indices_dict[counter] = np.fliplr(indices_dict[counter])
                counter += 1
                if ii > 0:
                    indices_temp = np.array(list(zip(range(np.max([height, width])), range(np.max([height, width])))))[
                                   ii:, :]
                    indices_temp[:, 0] -= ii
                    indices_dict[counter] = indices_temp[indices_temp[:, 0] < height,
                                            :] if height <= width else indices_temp[indices_temp[:, 1] < width, :]
                    slices_dict[counter] = ssdd_tensor[indices_dict[counter][:, 0], indices_dict[counter][:, 1], :]
                    # --------------------------------------------------------------------------------------------------
                    # Dealing with flipped input left -> right
                    # --------------------------------------------------------------------------------------------------
                    if fliplr:
                        indices_dict[counter][:, 1] = width - 1 - indices_dict[counter][:, 1]
                    # --------------------------------------------------------------------------------------------------
                    # Dealing with flipped input top -> bottom
                    # --------------------------------------------------------------------------------------------------
                    if flipud:
                        indices_dict[counter][:, 0] = height - 1 - indices_dict[counter][:, 0]
                    # --------------------------------------------------------------------------------------------------
                    # Dealing with transposed input
                    # --------------------------------------------------------------------------------------------------
                    if transpose:
                        indices_dict[counter] = np.fliplr(indices_dict[counter])
                    counter += 1

        return slices_dict, indices_dict

    def orient_direction_and_extract_slices(self, ssdd_tensor: np.ndarray, direction: int):
        """
        :param ssdd_tensor: 3D SSDD tensor with shape (H, W, label_size)
        :param direction: a number between 1 and 8:
            1. left      -> right
            2. top left  -> bot right
            3. top       -> bot
            4. top right -> bot left
            5. right     -> left
            6. bot right -> top left
            7. bot       -> top
            8. bot left  -> top right
        :return: The function orents the ssdd tensor such that the slices will be taken from left to right or across the
                 main diagonal. this means:
                     1. nothing
                     2. nothing
                     3. transpose
                     4. flip left -> right
                     5. flip left -> right
                     6. flip top -> bottom + flip left -> right
                     7. transpose + flip left -> right
                     8. flip top -> bottom
                 After performing the needed orientation change, the function calls the extract_slices method
        """
        if direction == 1:
            return self.extract_slices(ssdd_tensor, 1)
        elif direction == 2:
            return self.extract_slices(ssdd_tensor, 2)
        elif direction == 3:
            ssdd_tensor_transformed = np.moveaxis(ssdd_tensor, [0, 1, 2], [1, 0, 2])
            return self.extract_slices(ssdd_tensor_transformed, 1, transpose=True)
        elif direction == 4:
            ssdd_tensor_transformed = np.fliplr(ssdd_tensor)
            return self.extract_slices(ssdd_tensor_transformed, 2, fliplr=True)
        elif direction == 5:
            ssdd_tensor_transformed = np.fliplr(ssdd_tensor)
            return self.extract_slices(ssdd_tensor_transformed, 1, fliplr=True)
        if direction == 6:
            ssdd_tensor_transformed = np.fliplr(np.flipud(ssdd_tensor))
            return self.extract_slices(ssdd_tensor_transformed, 2, fliplr=True, flipud=True)
        elif direction == 7:
            ssdd_tensor_transformed = np.moveaxis(np.flipud(ssdd_tensor), [0, 1, 2], [1, 0, 2])
            return self.extract_slices(ssdd_tensor_transformed, 1, transpose=True, fliplr=True)
        elif direction == 8:
            ssdd_tensor_transformed = np.flipud(ssdd_tensor)
            return self.extract_slices(ssdd_tensor_transformed, 2, flipud=True)

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float,
                    direction: int = 1,
                    returnWhole: bool = False) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
            direction: integer depiction of the orientation of the dynamic programing. see documentation of
                       "orient_direction_and_extract_slices" API for more
            returnWhole: return l matrix (for all disparity values) or the final labeling
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        l = np.zeros_like(ssdd_tensor)
        slices_dict, indices_dict = self.orient_direction_and_extract_slices(ssdd_tensor, direction)
        # ==============================================================================================================
        # Running the forward MLSE computation in a for loop ONLY because this is requested in the exercise
        # ==============================================================================================================
        for ii in np.arange(0, len(slices_dict), 1):
            slice = slices_dict[ii]
            indices = indices_dict[ii]
            l[indices[:, 0], indices[:, 1]] = np.transpose(self.dp_grade_slice(np.squeeze(slice), p1, p2))
        # ==============================================================================================================
        # Labeling in the backward process of the MLSE
        # ==============================================================================================================
        if returnWhole:
            return l
        else:
            return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        for ii in range(num_of_directions):
            direction_to_slice[ii + 1] = self.dp_labeling(ssdd_tensor, p1, p2, direction=ii + 1)

        if False:
            import matplotlib.pyplot as plt
            from main_ours import load_data
            left_image, _ = load_data()
            fig, axes = plt.subplots(3, 3, sharex=True)
            axes[1, 0].imshow(direction_to_slice[1])
            axes[1, 0].set_title('Direction 1')
            axes[0, 0].imshow(direction_to_slice[2])
            axes[0, 0].set_title(f'Direction 2')
            axes[0, 1].imshow(direction_to_slice[3])
            axes[0, 1].set_title(f'Direction 3')
            axes[0, 2].imshow(direction_to_slice[4])
            axes[0, 2].set_title(f'Direction 4')
            axes[1, 2].imshow(direction_to_slice[5])
            axes[1, 2].set_title(f'Direction 5')
            axes[2, 2].imshow(direction_to_slice[6])
            axes[2, 2].set_title(f'Direction 6')
            axes[2, 1].imshow(direction_to_slice[7])
            axes[2, 1].set_title(f'Direction 7')
            axes[2, 0].imshow(direction_to_slice[8])
            axes[2, 0].set_title(f'Direction 8')
            axes[1, 1].imshow(left_image)
            axes[1, 1].set_title(f'Your Left Image')
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """

        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        for ii in range(num_of_directions):
            l += self.dp_labeling(ssdd_tensor, p1, p2, direction=ii + 1, returnWhole=True)
        l /= 8
        return self.naive_labeling(l)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Bonus functions
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    @staticmethod
    def get_gradient_map(image: np.ndarray) -> np.ndarray:
            # ==============================================================================================================
            # Local Variables
            # ==============================================================================================================
            kernel_sobel_horiz = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_sobel_verti = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            grad_map_horiz = np.zeros((image.shape[0], image.shape[1], 1))
            grad_map_verti = np.zeros((image.shape[0], image.shape[1], 1))
            laplacian_map = np.zeros((image.shape[0], image.shape[1], 1))
            for ii in range(image.shape[2]):
                grad_map_horiz[:, :, 0] += np.power(convolve2d(image[:, :, ii], kernel_sobel_horiz, mode='same'), 2)
                grad_map_verti[:, :, 0] += np.power(convolve2d(image[:, :, ii], kernel_sobel_verti, mode='same'), 2)
                laplacian_map[:, :, 0] += np.power(convolve2d(image[:, :, ii], kernel_laplacian, mode='same'), 2)

            return np.stack((grad_map_horiz, grad_map_verti, laplacian_map), axis=2)

    @staticmethod
    def dp_labeling_custom(ssdd_tensor: np.ndarray,
                           p1: float,
                           p2: float,
                           diagonals: bool = False) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
            diagonals: stating weather to compute diagonals or not
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        # ==============================================================================================================
        # Local Variables
        # ==============================================================================================================
        num_of_rows, num_of_cols, num_labels = ssdd_tensor.shape
        l               = np.zeros_like(ssdd_tensor)
        index_mat       = np.zeros_like(ssdd_tensor).astype(np.int)
        yy, xx          = np.meshgrid(np.arange(num_labels), np.arange(num_labels))
        depth_map       = np.zeros((num_of_rows, num_of_cols))

        left_indexing  = list(range(1,num_of_cols)) + [0]
        right_indexing = [num_of_cols-1] + list(range(0, num_of_cols-1))
        l_left = np.zeros_like(ssdd_tensor)
        l_right = np.zeros_like(ssdd_tensor)
        index_mat_left  = np.zeros_like(ssdd_tensor).astype(np.int)
        index_mat_right = np.zeros_like(ssdd_tensor).astype(np.int)
        depth_map_left  = np.zeros((num_of_rows, num_of_cols))
        depth_map_right = np.zeros((num_of_rows, num_of_cols))
        # ==============================================================================================================
        # Running the forward MLSE computation
        # ==============================================================================================================
        for row in np.arange(0, ssdd_tensor.shape[0], 1):
            if row == 0:
                l[row,:,:]         = ssdd_tensor[row,:,:]
                l_left[row, :, :]  = ssdd_tensor[row, :, :]
                l_right[row, :, :] = ssdd_tensor[row, :, :]
            else:
                # --------------------------------------------------------------------------------------------------
                # Extracting the state matrix
                # --------------------------------------------------------------------------------------------------
                state_matrix       = np.moveaxis(np.tile(ssdd_tensor[row, :, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                if diagonals:
                    state_matrix_left  = np.moveaxis(np.tile(ssdd_tensor[row, :, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                    state_matrix_right = np.moveaxis(np.tile(ssdd_tensor[row, :, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                # --------------------------------------------------------------------------------------------------
                # Computing the transition matrix
                # --------------------------------------------------------------------------------------------------
                # **********************************************************************************************
                # Filling initial values without penalties
                # **********************************************************************************************
                transition_matrix       = np.moveaxis(np.tile(l[row-1, :, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                if diagonals:
                    transition_matrix_left  = np.moveaxis(np.tile(l_left[row-1, left_indexing, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                    transition_matrix_right = np.moveaxis(np.tile(l_right[row-1, right_indexing, :], [num_labels, 1, 1]), [0, 1, 2], [1, 0, 2])
                # **********************************************************************************************
                # Adding P1 for deviation from the main diagonal by +-1
                # **********************************************************************************************
                transition_matrix[:,np.abs(yy - xx) == 1]       += p1
                if diagonals:
                    transition_matrix_left[:,np.abs(yy - xx) == 1]  += p1
                    transition_matrix_right[:,np.abs(yy - xx) == 1] += p1
                # **********************************************************************************************
                # Adding P2 for deviation from the main diagonal by +-2 or more
                # **********************************************************************************************
                transition_matrix[:,np.abs(yy - xx) >= 2]       += p2
                if diagonals:
                    transition_matrix_left[:,np.abs(yy - xx) >= 2]  += p2
                    transition_matrix_right[:,np.abs(yy - xx) >= 2] += p2
                # **********************************************************************************************
                # Summing
                # **********************************************************************************************
                tot_loss            = state_matrix + transition_matrix
                if diagonals:
                    tot_loss_left   = state_matrix_left + transition_matrix_left
                    tot_loss_right  = state_matrix_right + transition_matrix_right
                # --------------------------------------------------------------------------------------------------
                # Inserting the minimum value matching each label in the MLSE matrix
                # --------------------------------------------------------------------------------------------------
                l[row,:,:]= np.min(tot_loss, axis=2)
                # l[row,:,:]       = np.min(np.stack((np.min(tot_loss, axis=2),np.min(tot_loss_left, axis=2),np.min(tot_loss_right, axis=2)), axis=2), axis=2)
                if diagonals:
                    l_left[row,:,:]  = np.min(tot_loss_left, axis=2)
                    l_right[row,:,:] = np.min(tot_loss_right, axis=2)
                # --------------------------------------------------------------------------------------------------
                # Getting best path index
                # --------------------------------------------------------------------------------------------------
                index_mat[row-1, :]       = np.argmin(tot_loss, axis=2)
                if diagonals:
                    index_mat_left[row-1, :]  = np.argmin(tot_loss_left, axis=2)
                    index_mat_right[row-1, :] = np.argmin(tot_loss_right, axis=2)
        # ==============================================================================================================
        # Finished forward pass, filling depth map in the backward pass
        # ==============================================================================================================
        col_vec = np.arange(num_of_cols)
        index_vec       = np.argmin(l[-1,:,:], axis=1)
        if diagonals:
            index_vec_left  = np.argmin(l_left[-1,:,:], axis=1)
            index_vec_right = np.argmin(l_right[-1,:,:], axis=1)

        for row in np.arange(ssdd_tensor.shape[0]-1, 0, -1):
            depth_map[row,:]        = index_vec
            if diagonals:
                depth_map_left[row,:]   = index_vec_left
                depth_map_right[row,:]  = index_vec_right

            index_vec       = index_mat[row-1,col_vec,index_vec]
            if diagonals:
                index_vec_left  = index_mat_left[row-1,col_vec,index_vec_left]
                index_vec_right = index_mat_right[row-1,col_vec,index_vec_right]

        # return np.mean(np.stack((depth_map_left, depth_map, depth_map_right), axis=0), axis=0)
        if diagonals:
            return np.sum(np.stack((depth_map_left, depth_map, depth_map_right), axis=0), axis=0)
        else:
            return depth_map

    def orient_direction_and_label_custom(self, ssdd_tensor: np.ndarray,
                                          direction: int,
                                          p1: float,
                                          p2: float) ->np.ndarray:
        """
            :param ssdd_tensor: 3D SSDD tensor with shape (H, W, label_size)
            :param direction: a number between 1 and 4:
                                1. top       -> bottom
                                2. left      -> right
                                3. bottom    -> top
                                4. right     -> left
            :param p1: penalty for diff of 1
            :param p2: penalty of more than 2
            :return: The function orents the ssdd tensor such that the slices will be taken from left to right or across the
                     main diagonal. this means:
                         1. nothing
                         2. transpose
                         3. flip top  -> bottom
                         4. flip left -> right + transpose
                     After performing the needed orientation change, the function calls the dp_labeling_custom method
        """
        if direction == 1:
            return self.dp_labeling_custom(ssdd_tensor, p1, p2, diagonals=True)
        elif direction == 2:
            ssdd_tensor_transformed = np.moveaxis(ssdd_tensor, [0, 1, 2], [1, 0, 2])
            return np.transpose(self.dp_labeling_custom(ssdd_tensor_transformed, p1, p2))
        elif direction == 3:
            ssdd_tensor_transformed = np.flipud(ssdd_tensor)
            return np.flipud(self.dp_labeling_custom(ssdd_tensor_transformed, p1, p2, diagonals=True))
        elif direction == 4:
            ssdd_tensor_transformed = np.moveaxis(np.fliplr(ssdd_tensor), [0, 1, 2], [1, 0, 2])
            return np.transpose(np.flipud(self.dp_labeling_custom(ssdd_tensor_transformed, p1, p2)))

    def sgm_labeling_custom(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        kernel_size         = 7
        num_of_directions   = 4
        depth_map = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        for ii in range(num_of_directions):
            depth_map += self.orient_direction_and_label_custom(ssdd_tensor, direction=ii + 1, p1=p1, p2=p2)
        return medfilt2d(np.round(depth_map / (2*num_of_directions)), kernel_size=kernel_size).astype(np.int)

