import cv2
import matplotlib.image as mpimg

left_image = mpimg.imread('my_image_left_big.png')
right_image = mpimg.imread('my_image_right_big.png')

left_image_sampled = cv2.resize(src=left_image,dsize=(int(left_image.shape[1]/4),int(left_image.shape[0]/4)))
right_image_sampled = cv2.resize(src=right_image,dsize=(int(right_image.shape[1]/4),int(right_image.shape[0]/4)))

mpimg.imsave('my_image_left.png',left_image_sampled)
mpimg.imsave('my_image_right.png',right_image_sampled)
