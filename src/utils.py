import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def stretch_contrast(image):
    """
    Stretch the contrast of an image to utilize the full range of pixel values (0 to 255).
    
    Args:
    - image (numpy array): Input image represented as a numpy array.
    
    Returns:
    - stretched_image (numpy array): Image with contrast stretched to utilize the full range of pixel values.
    """
    # Compute the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Stretch the contrast using linear scaling
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
    return stretched_image.astype(np.uint8)

def load_data(base_path):
    images = []
    masks = []
    sample_names = []

    sample_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

    for sample_folder in sample_folders:
        sample_name = os.path.basename(sample_folder)
        image_path = os.path.join(sample_folder, 'images', f'{sample_name}.png')
        masks_path = os.path.join(sample_folder, 'masks')

        # Load the image and stretch contrast
        image = Image.open(image_path)
        image = np.array(image)
        stretched_image = stretch_contrast(image)
        images.append(stretched_image)
        sample_names.append(sample_name)

        # Load all masks for this sample
        mask_files = [os.path.join(masks_path, mask_file) for mask_file in os.listdir(masks_path) if mask_file.endswith('.png')]
        sample_masks = [np.array(Image.open(mask_file).convert('L')) for mask_file in mask_files]
        masks.append(sample_masks)

    return images, masks, sample_names

def merge_masks(masks):
    """
    Merge masks by adding the values of corresponding pixels.
    
    Args:
    - masks (list of numpy arrays): List of masks where each mask is represented as a numpy array.
    
    Returns:
    - merged_mask (numpy array): Merged mask obtained by adding the values of corresponding pixels in the input masks.
    """
    # Initialize the merged mask with zeros
    merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
    
    # Add the values of corresponding pixels in each mask
    for mask in masks:
        merged_mask += mask
    
    return merged_mask
