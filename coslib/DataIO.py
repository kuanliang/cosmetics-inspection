import numpy as np
import os
import cv2
import glob


IMAGE_SHAPE = (180, 180)
PIXEL_DEPTH = 256
path_to_images = './Data/Dataset-DetectNet_20161128_512-20170313T074149Z-001/Dataset-DetectNet_20161128_512/train/images/'

def load_images(path_to_images, shape=(256, 256)):
    '''load image from specified directory
    
    Args:
        path_to_images (string): 
        shape (tuple):
        
    return:
        dataset (numpy 3d array):
        
    Notes:
    
    '''
    # get image paths
    image_files = [x for x in os.listdir(path_to_images) if os.path.splitext(x)[1] == '.bmp']
    dataset = np.ndarray(shape=(len(image_files),
                                shape[0],
                                shape[1]),
                         dtype=np.float32)
    
    # fill in images into numpy array (tensor)
    for index, image_path in enumerate(glob.glob(path_to_images + '*.bmp')):
        #　print(image_path)
        rgb_image = cv2.imread(image_path)
        # print(rgb_image.shape)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        image_data = (gray_image.astype(float) - PIXEL_DEPTH) / PIXEL_DEPTH
        
        dataset[index,:,:] = image_data
    
    return dataset