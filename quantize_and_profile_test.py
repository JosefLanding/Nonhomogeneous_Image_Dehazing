import qai_hub as hub
import numpy as np
from PIL import Image
import os

# Compile the ONNX model for QCS8550 with NPU acceleration
compile_job = hub.submit_compile_job(
    model="dmphn_dehazing.onnx",  
    device=hub.Device("QCS8550 (Proxy)"),  
    options="--target_runtime onnx", 
    input_specs={"input": (1, 3, 600, 800)},  
)

assert isinstance(compile_job, hub.CompileJob)
compile_job.wait()
print("✅ Model compiled successfully for QCS8550 NPU!")

unquantized_onnx_model = compile_job.get_target_model()

# ImageNet mean and std (expected by many models)
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

# Set paths
images_dir = "hazy"  # Your folder with 50 hazy images
input_shape = (1, 3, 600, 800)  # Adjusted for your model (batch, channels, height, width)

sample_inputs = []

for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    
    # Open and preprocess image
    image = Image.open(image_path).convert("RGB").resize((input_shape[3], input_shape[2]))  # Resize to 800x600
    sample_input = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Convert to (C, H, W) and batch format
    sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
    
    # Normalize using mean and std
    sample_inputs.append(((sample_input - mean) / std).astype(np.float32))

# Prepare calibration data dictionary
calibration_data = dict(input=sample_inputs)

print(f"Loaded {len(sample_inputs)} calibration images.")

quantize_job = hub.submit_quantize_job(
    model=unquantized_onnx_model,
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)

quantized_onnx_model = quantize_job.get_target_model()
assert isinstance(quantized_onnx_model, hub.Model)


# 6. Compile to target runtime (TFLite)
compile_onnx_job = hub.submit_compile_job(
    model=quantized_onnx_model,
     device=hub.Device("QCS8550 (Proxy)"),  
    options="--target_runtime tflite --quantize_io",
)
assert isinstance(compile_onnx_job, hub.CompileJob)

profile_job = hub.submit_profile_job(
    model=compile_onnx_job.get_target_model(),
    device=hub.Device("QCS8550 (Proxy)"),
)
assert isinstance(profile_job, hub.ProfileJob)


# Path to your input PNG image
image_path = "test800.png"  # Adjust the path as needed

# Step 1: Load the image using PIL
image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already

# Step 2: Preprocess the image
# Resize to the expected input size (1600x1200) and normalize if needed
#image = image.resize((1200, 1600))  # Resize image to match the input shape (H, W)
#image.show()

# Convert the image to a numpy array
image = np.array(image, dtype=np.float32)  # Convert the image to a numpy array

# Normalize the image by subtracting 0.5 (as done in the local code)
image = (image / 255.0) - 0.5  # Normalize to range [-0.5, 0.5]

# Reorder the dimensions to match (Batch, Channels, Height, Width)
image = np.transpose(image, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)

# Step 3: Submit inference job using the compiled model
inference_job = hub.submit_inference_job(
    model=compile_onnx_job.get_target_model(),
    device=hub.Device("QCS8550 (Proxy)"),
    inputs={"input": [image]},  # Pass the preprocessed image
)

# Step 4: Download the output data from the inference job
inference_results = inference_job.download_output_data()

output_name = list(inference_results.keys())[0]
output_data = inference_results[output_name][0]

print("Output data shape:", output_data.shape)

# Assuming the output has a batch dimension (e.g., batch_size, channels, height, width)
# Extract the first image if there's a batch dimension
output_image = output_data[0]  # Extract the first image from the batch

# Now you can transpose the image to (H, W, C) format
output_image = np.transpose(output_image, (1, 2, 0))

# Rescale to [0, 255] if needed and convert back to a PIL image
output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

# Save or display the result
output_image = Image.fromarray(output_image)  # Convert back to a PIL image
#output_image = output_image.resize((1600, 1200))

# Save or display the result
output_image.save("inferenceResults/dehazed_result.png")  # Save the result
output_image.show()  # Optionally display the image