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



def apply_physics_based_haze3(image_path, output_path):
    """
    Apply haze using an approximation of Koschmieder’s law.
    """
    # Load and normalize the image (ONLY ONCE)
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    height, width, _ = image.shape

    # Generate depth approximation using Perlin noise
    depth = generate_perlin_noise2((height, width)).astype(np.float32) / 255.0  # Ensure it's in range [0, 1]

    # Haze parameters
    beta = random.uniform(2.0, 2.7)  # Scattering coefficient
    A = 0.8  # Atmospheric light intensity

    # Compute transmission map
    transmission = np.exp(-beta * depth)

    # Apply haze model (Ensure values remain in [0, 1])
    hazy_image = A * (1 - transmission[..., np.newaxis]) + (image * transmission[..., np.newaxis])
    hazy_image = np.clip(hazy_image, 0, 1)  # Ensure values stay within [0,1]

    # Convert back to 8-bit and save
    hazy_image = (hazy_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, hazy_image)
    print(f"Hazy image saved: {output_path}")


apply_physics_based_haze3("test_apply_haze.png", "hazy_drone_image_physics2.png")

# Define paths
# source_folder = r"C:\Users\josef\Downloads\archive"
# dataset_folder = r"dataset"
# gt_folder = os.path.join(dataset_folder, "GT")
# hazy_folder = os.path.join(dataset_folder, "hazy")

# # Create folders if they don't exist
# os.makedirs(gt_folder, exist_ok=True)
# os.makedirs(hazy_folder, exist_ok=True)


# image_files = [f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# for i, image_name in enumerate(tqdm(image_files, desc="Processing Images"), start=1):
#     # Load and resize image
#     image_path = os.path.join(source_folder, image_name)
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (800, 600))

#     # Generate sequential filename
#     filename = f"{i}.jpg"

#     # Save original to GT folder
#     gt_path = os.path.join(gt_folder, filename)
#     cv2.imwrite(gt_path, image)

#     # Apply haze and save to hazy folder
#     hazy_image = apply_physics_based_haze2(image)
#     hazy_path = os.path.join(hazy_folder, filename)
#     cv2.imwrite(hazy_path, hazy_image)

# print("✅ Dataset preparation complete! GT and hazy images saved.")


# def apply_physics_based_haze2(image_path,output_path):
#     """
#     Apply haze using an approximation of Koschmieder’s law.
#     """
#     image = cv2.imread(image_path).astype(np.float32) / 255.0
#     image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
#     #image = image.astype(np.float32) / 255.0  # Normalize
#     height, width, _ = image.shape
#     #height, width, _ = image.shape

#     # Generate depth approximation using Perlin noise
#     depth = generate_perlin_noise2((height, width)) / 255.0
#     beta = random.uniform(2.0, 2.7) # 2.7  # Scattering coefficient (higher = more haze)
#     A =  0.8  # Atmospheric light intensity

#     # Compute transmission map
#     transmission = np.exp(-beta * depth)
    
#     # Apply haze model
#     hazy_image = A * (1 - transmission[..., np.newaxis]) + (image * transmission[..., np.newaxis])


#     # Convert back to 8-bit and save
#     #return (hazy_image * 255).astype(np.uint8)
#     cv2.imwrite(output_path, hazy_image)
#     print(f"Hazy image saved: {output_path}")



# def apply_physics_based_haze(input_image_path, output_image_path):
#     # Load and resize image
#     image = Image.open(input_image_path).convert("RGB")
#     image = image.resize((800, 600), Image.BICUBIC)
#     image = np.array(image) / 255.0  # Normalize to [0,1]

#     # Generate nonhomogeneous transmission map
#     transmission = generate_nonhomogeneous_transmission(600, 800, scale=50)

#     # Add small random noise for realism
#     random_noise = np.random.uniform(-0.05, 0.05, (600, 800))
#     transmission = np.clip(transmission + random_noise, 0.3, 1.0)  # Keep in valid range

#     # Expand transmission to (600,800,3) for broadcasting
#     transmission = transmission[..., np.newaxis]

#     # Atmospheric light (A) - Set to white light
#     A = np.array([1.0, 1.0, 1.0])

#     # Apply haze effect
#     hazy_image = A * (1 - transmission) + (image * transmission)

#     # Convert to uint8 format
#     hazy_image = (hazy_image * 255).astype(np.uint8)
    
#     # Save the output hazy image
#     Image.fromarray(hazy_image).save(output_image_path)

# Example usage
#apply_physics_based_haze("test_apply_haze.png", "hazy_drone_image_nonhomogeneous.png")

#apply_physics_based_haze("test_apply_haze.png", "hazy_drone_image_physics.png")



# def apply_physics_based_haze(input_image_path, output_image_path):
#     # Load the image
#     image = Image.open(input_image_path).convert("RGB")
    
#     # Resize image to 800x600
#     image = image.resize((800, 600), Image.BICUBIC)
#     image = np.array(image) / 255.0  # Normalize to [0, 1]

#     # Generate a synthetic transmission map (Example: smooth gradient)
#     transmission = np.linspace(0.3, 1, 600).reshape(600, 1)  # Vertical gradient
#     transmission = np.repeat(transmission, 800, axis=1)  # Expand horizontally
    
#     # Resize transmission to match (800,600) shape
#     transmission = cv2.resize(transmission, (800, 600), interpolation=cv2.INTER_LINEAR)

#     # Expand transmission to (800,600,3) for broadcasting
#     transmission = transmission[..., np.newaxis]

#     # Atmospheric light (A) assumed to be white
#     A = np.array([1.0, 1.0, 1.0])  

#     # Apply the haze effect
#     hazy_image = A * (1 - transmission) + (image * transmission)

#     # Convert back to uint8 image format
#     hazy_image = (hazy_image * 255).astype(np.uint8)
    
#     # Save the output hazy image
#     Image.fromarray(hazy_image).save(output_image_path)


# def apply_haze(image_path, output_path):
#     """
#     Apply synthetic non-homogeneous haze using Perlin noise.
#     """
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (800, 600))  # Resize if necessary
#     haze_layer = generate_perlin_noise(image.shape[:2])

#     # Convert to 3-channel grayscale haze
#     haze_layer = cv2.merge([haze_layer, haze_layer, haze_layer])

#     # Blend original image with haze
#     alpha = 0.6  # Adjust haze intensity
#     hazy_image = cv2.addWeighted(image, 1 - alpha, haze_layer, alpha, 0)

#     cv2.imwrite(output_path, hazy_image)
#     print(f"Hazy image saved: {output_path}")


# Example usage
#apply_haze("test_apply_haze.png", "hazy_drone_image.png")


# def generate_perlin_noise(shape=(600, 800), scale=100):
#     """
#     Generate Perlin noise for non-homogeneous haze simulation.
#     """
#     height, width = shape
#     perlin_noise = np.zeros((height, width))

#     for i in range(height):
#         for j in range(width):
#             perlin_noise[i][j] = noise.pnoise2(i / scale, j / scale, octaves=6)

#     # Normalize to 0-255
#     perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
#     perlin_noise = (perlin_noise * 255).astype(np.uint8)
    
#     return perlin_noise