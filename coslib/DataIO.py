import numpy as np
import os
import cv2
import glob


IMAGE_SHAPE = (180, 180)
PIXEL_DEPTH = 256
path_to_images = './Data/Dataset-DetectNet_20161128_512-20170313T074149Z-001/Dataset-DetectNet_20161128_512/train/images/'

def load_images(path_to_images, shape=(256, 256), filename_index=True):
    '''load image from specified directory
    
    Args:
        path_to_images (string): 
        shape (tuple):
        
    return:
        dataset (numpy 3d array):
        
    Notes:
    
    '''
    # get image paths
    image_files = [x for x in os.listdir(path_to_images) if (os.path.splitext(x)[1] == '.bmp') or
                                                            (os.path.splitext(x)[1] == '.png')]
    dataset = np.ndarray(shape=(len(image_files),
                                shape[0],
                                shape[1]),
                         dtype=np.float32)
    
    # fill in images into numpy array (tensor)
    file_ext = os.path.splitext(image_files[0])[1]
    
    # print(file_ext)
    for index, image_path in enumerate(glob.glob(path_to_images + '*{}'.format(file_ext))):
        # print(image_path)
        rgb_image = cv2.imread(image_path)
        # print(rgb_image.shape)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        image_data = (gray_image.astype(float) - PIXEL_DEPTH) / PIXEL_DEPTH
        
        if not filename_index:
            dataset[index,:,:] = image_data
        else:
            file_index = int(os.path.splitext(os.path.basename(image_path))[0]) - 1
            # print(file_index)
            dataset[file_index,:,:] = image_data
            # print(file_index)
    return dataset



def load_coordinates(path_to_coor):
    '''
    '''
    
    coord_dict = {}
    coord_dict_all = {}
    with open(path_to_coor) as f:
        coordinates = f.read().split('\n')
        for coord in coordinates:
            #print(len(coord.split('\t')))
            if len(coord.split('\t')) == 6:
                coord_dict = {}
                coord_split = coord.split('\t')
                # print(coord_split)
                # print('\n')
                coord_dict['major_axis'] = round(float(coord_split[1]))
                coord_dict['minor_axis'] = round(float(coord_split[2]))
                coord_dict['angle'] = float(coord_split[3])
                coord_dict['x'] = round(float(coord_split[4]))
                coord_dict['y'] = round(float(coord_split[5]))
                index = int(coord_split[0]) - 1
                coord_dict_all[index] = coord_dict
    
    return coord_dict_all
