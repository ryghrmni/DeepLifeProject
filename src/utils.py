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
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
    return stretched_image.astype(np.uint8)

def resize_image(image, size=(256, 256)):
    """
    Resize the input image to the specified size.
    
    Args:
    - image (PIL.Image or numpy array): Input image.
    - size (tuple of ints): Desired output size (width, height).
    
    Returns:
    - resized_image (PIL.Image): Resized image.
    """
    resize_transform = transforms.Resize(size)
    return resize_transform(image)


def normalize_image(image, mean=0.5, std=0.5):
    """
    Normalize the input image using the specified mean and standard deviation.
    
    Args:
    - image (torch.Tensor): Input image tensor.
    - mean (float): Mean for normalization.
    - std (float): Standard deviation for normalization.
    
    Returns:
    - normalized_image (torch.Tensor): Normalized image tensor.
    """
    normalize_transform = transforms.Normalize(mean=[mean], std=[std])
    return normalize_transform(image)


def standardize_image(image):
    """
    Standardize the input image by subtracting the mean and dividing by the standard deviation.
    
    Args:
    - image (PIL.Image or numpy array): Input image.
    
    Returns:
    - standardized_image (PIL.Image): Standardized image.
    """
    image_array = np.array(image)
    mean, std = image_array.mean(), image_array.std()
    standardized_image = (image_array - mean) / std
    return Image.fromarray(standardized_image.astype(np.uint8))

def equalize_histogram(image):
    """
    Apply histogram equalization to the input image to enhance contrast.
    
    Args:
    - image (PIL.Image or numpy array): Input image.
    
    Returns:
    - equalized_image (PIL.Image): Image with equalized histogram.
    """
    image_array = np.array(image)
    equalized_image = cv2.equalizeHist(image_array)
    return Image.fromarray(equalized_image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.
    
    Args:
    - image (PIL.Image or numpy array): Input image.
    - clip_limit (float): Threshold for contrast limiting.
    - tile_grid_size (tuple of ints): Size of the grid for histogram equalization.
    
    Returns:
    - clahe_image (PIL.Image): Image after applying CLAHE.
    """
    image_array = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image_array)
    return Image.fromarray(clahe_image)


# Should be Modified for augmenting both image and mask with same properties for each sample
def augment_image(image, size=(256, 256)):
    """
    Apply random augmentations to the input image for data augmentation.
    
    Args:
    - image (PIL.Image): Input image.
    - size (tuple of ints): Desired output size (width, height).
    
    Returns:
    - augmented_image (PIL.Image): Augmented image.
    """
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
    ])
    return augmentation_transform(image)

def to_tensor(image):
    """
    Convert the input image to a tensor.
    
    Args:
    - image (PIL.Image): Input image.
    
    Returns:
    - tensor_image (torch.Tensor): Image converted to a tensor.
    """
    tensor_transform = transforms.ToTensor()
    return tensor_transform(image)

def preprocess_image(image, size=(256, 256), mean=0.5, std=0.5, apply_clahe=False, apply_equalization=False, apply_standardization=False):
    """
    Preprocess the input image with optional CLAHE, histogram equalization, and standardization.
    
    Args:
    - image (PIL.Image): Input image.
    - size (tuple of ints): Desired output size (width, height).
    - mean (float): Mean for normalization.
    - std (float): Standard deviation for normalization.
    - apply_clahe (bool): Whether to apply CLAHE.
    - apply_equalization (bool): Whether to apply histogram equalization.
    - apply_standardization (bool): Whether to apply standardization.
    
    Returns:
    - preprocessed_image (torch.Tensor): Preprocessed image tensor.
    """
    image = resize_image(image, size)
    if apply_clahe:
        image = apply_clahe(image)
    if apply_equalization:
        image = equalize_histogram(image)
    if apply_standardization:
        image = standardize_image(image)
    image = normalize_image(to_tensor(image), mean, std)
    return image

class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        """
        Custom dataset for loading images and masks.
        
        Args:
        - images (list of numpy arrays): List of images.
        - masks (list of numpy arrays): List of corresponding masks.
        - transform (callable, optional): A function/transform to apply to the images.
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx][0])  # Assuming single mask per image
        if self.transform:
            image = self.transform(image)
        return image, mask

def load_data(base_path):
    """
    Load images and masks from the specified base path.
    
    Args:
    - base_path (str): Path to the dataset.
    
    Returns:
    - images (list of numpy arrays): List of preprocessed images.
    - masks (list of lists of numpy arrays): List of corresponding masks for each image.
    - sample_names (list of str): List of sample names.
    """
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
        images.append(image)
        # stretched_image = stretch_contrast(image) --> should be tested
        # images.append(stretched_image)
        sample_names.append(sample_name)

        # Load all masks for this sample
        mask_files = [os.path.join(masks_path, mask_file) for mask_file in os.listdir(masks_path) if mask_file.endswith('.png')]
        sample_masks = [np.array(Image.open(mask_file).convert('L')) for mask_file in mask_files]
        masks.append(sample_masks)

    return images, masks, sample_names


def create_data_loader(base_path, batch_size=16, shuffle=True, size=(256, 256), mean=0.5, std=0.5, apply_clahe=False, apply_equalization=False, apply_standardization=False):
    """
    Create a DataLoader for batching and shuffling the preprocessed image and mask data.
    
    Args:
    - base_path (str): Path to the dataset.
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the data.
    - size (tuple of ints): Desired output size (width, height).
    - mean (float): Mean for normalization.
    - std (float): Standard deviation for normalization.
    - apply_clahe (bool): Whether to apply CLAHE.
    - apply_equalization (bool): Whether to apply histogram equalization.
    - apply_standardization (bool): Whether to apply standardization.
    
    Returns:
    - data_loader (DataLoader): DataLoader for the dataset.
    """
    images, masks, _ = load_data(base_path)
    dataset = CustomDataset(images, masks, transform=lambda img: preprocess_image(img, size, mean, std, apply_clahe, apply_equalization, apply_standardization))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



# -- Usage of DataLoader --
# from utils import create_data_loader

# # Define the path to your dataset
# base_path = 'path_to_your_dataset'

# # Define DataLoader parameters
# batch_size = 16
# shuffle = True
# size = (256, 256)
# mean = 0.5
# std = 0.5
# apply_clahe = True
# apply_equalization = False
# apply_standardization = True

# # Create DataLoader
# data_loader = create_data_loader(
#     base_path=base_path,
#     batch_size=batch_size,
#     shuffle=shuffle,
#     size=size,
#     mean=mean,
#     std=std,
#     apply_clahe=apply_clahe,
#     apply_equalization=apply_equalization,
#     apply_standardization=apply_standardization
# )

# # Iterate through the DataLoader
# for images, masks in data_loader:
#     # Process the batch of images and masks
#     # images and masks are tensors of shape (batch_size, channels, height, width)
#     print("Batch of images shape:", images.shape)
#     print("Batch of masks shape:", masks.shape)
#     # Your training/evaluation code here...

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
