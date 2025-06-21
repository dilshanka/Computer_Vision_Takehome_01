import cv2
import numpy as np
import os


input_path = "../images/input.jpg" 
output_dir = "../outputs"           
os.makedirs(output_dir, exist_ok=True)


img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)


# Intensity Reduction
def reduce_intensity_levels(image, levels):
    factor = 256 // levels
    reduced = (image // factor) * factor
    return reduced

# Spatial Averaging
def spatial_average(image, ksize):
    return cv2.blur(image, (ksize, ksize))

# Rotation
def rotate_image(image, angle):
    (h, w) = image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

# Block Averaging
def block_average(image, block_size):
    h, w = image.shape
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    output = image[:h_crop, :w_crop].copy()
    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = output[i:i+block_size, j:j+block_size]
            avg = int(np.mean(block))
            output[i:i+block_size, j:j+block_size] = avg
    return output



# 1. Intensity Reduction to 2 levels
cv2.imwrite(f"{output_dir}/reduced_2_levels.png", reduce_intensity_levels(img, 2))

# 2. Spatial Averages
cv2.imwrite(f"{output_dir}/avg_3x3.png", spatial_average(img, 3))
cv2.imwrite(f"{output_dir}/avg_10x10.png", spatial_average(img, 10))
cv2.imwrite(f"{output_dir}/avg_20x20.png", spatial_average(img, 20))

# 3. Rotation
cv2.imwrite(f"{output_dir}/rotated_45.png", rotate_image(img, 45))
cv2.imwrite(f"{output_dir}/rotated_90.png", rotate_image(img, 90))

# 4. Block Averages
cv2.imwrite(f"{output_dir}/block_avg_3x3.png", block_average(img, 3))
cv2.imwrite(f"{output_dir}/block_avg_5x5.png", block_average(img, 5))
cv2.imwrite(f"{output_dir}/block_avg_7x7.png", block_average(img, 7))


