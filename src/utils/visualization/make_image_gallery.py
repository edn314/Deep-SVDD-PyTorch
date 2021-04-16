import torch
# import matplotlib
# matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os
import random
from PIL import Image, ImageOps

def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
              for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col  
            image.paste(images[idx], offset)
    
    return image

if __name__ == "__main__":
    # Get list of image paths
    folder = '/n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/end-to-end-kmeans-models-no-scaling/multi-center-naive/images_for_gallery'
    image_paths = [os.path.join(folder, f) 
                for f in os.listdir(folder) if f.endswith('.png')]

    # Random selection of images
    image_array = random.sample(image_paths, k=128)

    # Create and save image grid
    image = concat_images(image_array, (100, 100), (16, 8))
    image.save('gallery_128.png', 'PNG')
