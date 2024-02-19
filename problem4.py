from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
import numpy as np


def manual_otsu_threshold(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    pixel_counts, pixel_values = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    current_max, threshold = 0, 0
    sum_total = np.cumsum(pixel_counts * np.arange(256))

    weight_background = 0
    sum_background = 0
    for i in range(256):
        weight_background += pixel_counts[i]
        sum_background += i * pixel_counts[i]
        weight_foreground = total_pixels - weight_background
        if weight_background == 0 or weight_foreground == 0:
            continue
        sum_foreground = sum_total[-1] - sum_background
        mean_background = sum_background / weight_background
        mean_foreground = sum_foreground / weight_foreground
        between_class_variance = (weight_background * weight_foreground * 
                                  (mean_background - mean_foreground) ** 2)
        
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    return threshold

def segment_image_otsu_and_save(input_image_path, output_image_path):
    image = imread(input_image_path, as_gray=True)

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    otsu_threshold = manual_otsu_threshold(image)
    segmented_image = image > otsu_threshold
    segmented_image_uint8 = (segmented_image * 255).astype('uint8')
    imsave(output_image_path, segmented_image_uint8)
    return otsu_threshold



# Define the paths for input and output
input_image_path = './Examples/ImageB3.JPG'
output_image_path = './outputP4.jpg'

# Apply the function and save the segmented image
otsu_threshold_value = segment_image_otsu_and_save(input_image_path, output_image_path)
print(f"The Otsu's threshold value is: {otsu_threshold_value}")


