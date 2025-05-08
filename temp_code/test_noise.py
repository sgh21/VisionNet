import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

# 添加项目根目录到 Python 路径
sys.path.append('/home/sgh/data/WorkSpace/VisionNet')

# 从 LocalMAE 导入 add_zero_noise 函数
from models.LocalMAE import LocalMAE

def load_images_from_folder(folder_path, max_images=4):
    """
    Load multiple images from the specified folder
    
    Args:
        folder_path: Path to the image folder
        max_images: Maximum number of images to load
        
    Returns:
        List of PIL Image objects
    """
    images = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # List all files in the folder
    files = os.listdir(folder_path)
    # Only keep valid image files
    image_files = [f for f in files if os.path.splitext(f.lower())[1] in valid_extensions]
    
    # Randomly select at most max_images
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)
    
    # Load images
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            img = Image.open(file_path).convert('RGB')
            images.append(img)
            print(f"Loaded image: {file_name}")
        except Exception as e:
            print(f"Failed to load image {file_name}: {e}")
            
    return images

def process_images(images, size=(224, 224)):
    """
    Process a list of images, converting them to a batch tensor
    
    Args:
        images: List of PIL Image objects
        size: Target size for resizing
        
    Returns:
        torch.Tensor: Image tensor of shape [B, C, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    
    # Convert each image to a tensor and stack them
    tensor_list = [transform(img) for img in images]
    batch_tensor = torch.stack(tensor_list)
    
    return batch_tensor

def denormalize_for_display(tensor):
    """
    Convert the tensor to a numpy array suitable for display
    
    Args:
        tensor: Image tensor in range [0, 1]
        
    Returns:
        numpy array in range [0, 1]
    """
    # Clone the tensor to avoid modifying the original
    img = tensor.clone().detach()
    
    # No need for denormalization if already in [0, 1]
    # Just ensure values are clipped to valid range
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy array in HWC format
    img_np = img.permute(1, 2, 0).numpy()
    
    return img_np

def test_noise_level():
    """Test different noise levels with fixed ratio=1.0"""
    # Create a LocalMAE model instance
    model = LocalMAE(
        img_size=224,
        patch_size=16,
        in_chans=3, 
        embed_dim=768,
        depth=1,
        encoder_num_heads=12,
        cross_num_heads=8,
        feature_dim=3
    )
    
    # Load images from the data directory
    data_dir = '/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/data_all/rgb_images'
    images = load_images_from_folder(data_dir)
    
    if not images:
        print("No images found. Please check the path.")
        return
    
    # Convert images to batch tensor
    batch_tensor = process_images(images)
    print(f"Image tensor shape: {batch_tensor.shape}")
    
    # Fixed noise ratio
    noise_ratio = 1.0
    
    # Test different noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Organize noise levels into two rows for better visualization
    first_row_levels = noise_levels[:3]  # [0.05, 0.1, 0.2]
    second_row_levels = noise_levels[3:] # [0.3, 0.4, 0.5]
    
    # Create subplots - two rows with original image + different noise levels
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows x 4 columns (original + 3 levels per row)
    
    # Display original image in both rows
    for row in range(2):
        axes[row, 0].imshow(denormalize_for_display(batch_tensor[0]))
        axes[row, 0].set_title("Original Image")
        axes[row, 0].axis('off')
    
    # Test first row of noise levels
    for i, level in enumerate(first_row_levels):
        # Add noise with fixed ratio=1.0
        noisy_tensor = model.add_zero_noise(batch_tensor, noise_ratio=noise_ratio, noise_level=level)
        
        # Display the result
        noisy_img_np = denormalize_for_display(noisy_tensor[0])
        axes[0, i+1].imshow(noisy_img_np)
        axes[0, i+1].set_title(f"Noise Level={level}")
        axes[0, i+1].axis('off')
    
    # Test second row of noise levels
    for i, level in enumerate(second_row_levels):
        # Add noise with fixed ratio=1.0
        noisy_tensor = model.add_zero_noise(batch_tensor, noise_ratio=noise_ratio, noise_level=level)
        
        # Display the result
        noisy_img_np = denormalize_for_display(noisy_tensor[0])
        axes[1, i+1].imshow(noisy_img_np)
        axes[1, i+1].set_title(f"Noise Level={level}")
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/sgh/data/WorkSpace/VisionNet/temp_code/noise_level_test.png', dpi=150)
    plt.show()
    
    # Test effect on multiple images
    print("\nTesting noise effect on batch...")
    selected_level = 0.3  # Select a specific noise level for batch testing
    
    # Apply noise to the entire batch
    noisy_batch = model.add_zero_noise(batch_tensor, noise_ratio=noise_ratio, noise_level=selected_level)
    
    # Create visualization for all images
    fig, axes = plt.subplots(2, len(batch_tensor), figsize=(len(batch_tensor) * 4, 8))
    
    # Display original and noisy images
    for i in range(len(batch_tensor)):
        # Original image
        orig_img_np = denormalize_for_display(batch_tensor[i])
        axes[0, i].imshow(orig_img_np)
        axes[0, i].set_title(f"Original Image {i+1}")
        axes[0, i].axis('off')
        
        # Noisy image
        noisy_img_np = denormalize_for_display(noisy_batch[i])
        axes[1, i].imshow(noisy_img_np)
        axes[1, i].set_title(f"Noisy Image {i+1} (Level={selected_level})")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/sgh/data/WorkSpace/VisionNet/temp_code/noise_batch_test.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    test_noise_level()