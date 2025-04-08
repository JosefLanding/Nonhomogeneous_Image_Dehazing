import cv2
import numpy as np
from PIL import Image
import noise
import random
import os
from tqdm import tqdm


def generate_perlin_noise2(shape):
    """Generate a Perlin noise map with randomized parameters."""
    height, width = shape
    
    # Randomize parameters for variation
    scale = random.uniform(100, 200)  # Random scale between 100 and 200
    octaves = random.randint(4, 8)  # Random octaves between 4 and 8
    persistence = random.uniform(0.3, 0.7)  # Controls amplitude variation
    lacunarity = random.uniform(1.5, 3.0)  # Controls frequency increase

    perlin_noise = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            perlin_noise[i, j] = noise.pnoise2(
                i / scale, j / scale, 
                octaves=octaves, 
                persistence=persistence, 
                lacunarity=lacunarity
            )

    # Normalize to 0-255
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min()) * 255
    return perlin_noise.astype(np.uint8)





def apply_physics_based_haze2(image):
    """
    Apply haze using an approximation of Koschmieder’s law.
    """
    # image = cv2.imread(image_path).astype(np.float32) / 255.0
    # image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0  # Normalize
    height, width, _ = image.shape
    #height, width, _ = image.shape

    # Generate depth approximation using Perlin noise
    depth = generate_perlin_noise2((height, width)) / 255.0
    beta = random.uniform(2.0, 2.7) # 2.7  # Scattering coefficient (higher = more haze)
    A =  0.8  # Atmospheric light intensity

    # Compute transmission map
    transmission = np.exp(-beta * depth)
    
    # Apply haze model
    hazy_image = A * (1 - transmission[..., np.newaxis]) + (image * transmission[..., np.newaxis])


    # Convert back to 8-bit and save
    return (hazy_image * 255).astype(np.uint8)
    #cv2.imwrite(output_path, hazy_image)
    #print(f"Hazy image saved: {output_path}")




# Define paths
source_folder = r"VSAI_Data"
dataset_folder = r"dataset"
gt_folder = os.path.join(dataset_folder, "GT")
hazy_folder = os.path.join(dataset_folder, "hazy")

# Create folders if they don't exist
os.makedirs(gt_folder, exist_ok=True)
os.makedirs(hazy_folder, exist_ok=True)


image_files = [f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

for i, image_name in enumerate(tqdm(image_files, desc="Processing Images"), start=1):
    # Load and resize image
    image_path = os.path.join(source_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))

    # Generate sequential filename
    filename = f"{i}.jpg"

    # Save original to GT folder
    gt_path = os.path.join(gt_folder, filename)
    cv2.imwrite(gt_path, image)

    # Apply haze and save to hazy folder
    hazy_image = apply_physics_based_haze2(image)
    hazy_path = os.path.join(hazy_folder, filename)
    cv2.imwrite(hazy_path, hazy_image)

print("✅ Dataset preparation complete! GT and hazy images saved.")
