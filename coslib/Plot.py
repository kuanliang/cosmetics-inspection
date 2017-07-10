import cv2
import matplotlib.pyplot as plt

def plot_defect(raw_img, coordinates):
    
    img_copy = raw_img.copy()
    
    # return img_copy
    
    return_img = cv2.ellipse(img_copy,
                             (coordinates['x'], coordinates['y']),
                             (coordinates['major_axis'], coordinates['minor_axis']),
                             coordinates['angle'] * 90,
                             # -90,
                             0,
                             360,
                             (0, 0, 0),
                             2)
    
    plt.imshow(return_img, cmap='gray')
    print('angle: {}'.format(coordinates['angle']))
    print('angle(degree): {}'.format(90 * coordinates['angle']))
    
    
    return return_img