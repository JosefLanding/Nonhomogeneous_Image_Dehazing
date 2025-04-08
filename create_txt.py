# import os

# dataset_path = "dataset"
# hazy_path = os.path.join(dataset_path, "hazy")
# gt_path = os.path.join(dataset_path, "GT")

# hazy_images = sorted(os.listdir(hazy_path))
# gt_images = sorted(os.listdir(gt_path))

# with open(os.path.join(dataset_path, "hazy.txt"), "w") as f:
#     for img in hazy_images:
#         f.write(f"hazy/{img}\n")  # Save relative path

# with open(os.path.join(dataset_path, "GT.txt"), "w") as f:
#     for img in gt_images:
#         f.write(f"GT/{img}\n")  # Save relative path

# print("✅ Image list files created!")


import os
import random
import shutil

dataset_path = "dataset"
hazy_path = os.path.join(dataset_path, "hazy")
gt_path = os.path.join(dataset_path, "GT")

# Create necessary directories
os.makedirs(hazy_path, exist_ok=True)
os.makedirs(gt_path, exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val_hazy"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val_GT"), exist_ok=True)

# List all images in hazy and GT directories
hazy_images = sorted(os.listdir(hazy_path))
gt_images = sorted(os.listdir(gt_path))

# Ensure both directories have the same number of images
assert len(hazy_images) == len(gt_images), "The number of hazy and GT images must be the same."

# Set number of validation images (1000)
val_size = 1000
train_size = len(hazy_images) - val_size

# Split into train and validation sets (randomly shuffle the images)
train_hazy_images = hazy_images[val_size:]
train_gt_images = gt_images[val_size:]

val_hazy_images = hazy_images[:val_size]
val_gt_images = gt_images[:val_size]

# Move the images to the respective directories
for img in train_hazy_images:
    shutil.move(os.path.join(hazy_path, img), os.path.join(dataset_path, "hazy", img))

for img in train_gt_images:
    shutil.move(os.path.join(gt_path, img), os.path.join(dataset_path, "GT", img))

for img in val_hazy_images:
    shutil.move(os.path.join(hazy_path, img), os.path.join(dataset_path, "val_hazy", img))

for img in val_gt_images:
    shutil.move(os.path.join(gt_path, img), os.path.join(dataset_path, "val_GT", img))

# Create the .txt files for training and validation
with open(os.path.join(dataset_path, "hazy.txt"), "w") as f:
    for img in train_hazy_images:
        f.write(f"hazy/{img}\n")  # Save relative path

with open(os.path.join(dataset_path, "GT.txt"), "w") as f:
    for img in train_gt_images:
        f.write(f"GT/{img}\n")  # Save relative path

with open(os.path.join(dataset_path, "val_hazy.txt"), "w") as f:
    for img in val_hazy_images:
        f.write(f"val_hazy/{img}\n")  # Save relative path

with open(os.path.join(dataset_path, "val_GT.txt"), "w") as f:
    for img in val_gt_images:
        f.write(f"val_GT/{img}\n")  # Save relative path

print("✅ Dataset preparation complete! Folders and .txt files created!")
