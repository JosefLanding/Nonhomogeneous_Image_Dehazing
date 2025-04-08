import qai_hub as hub
import numpy as np
from PIL import Image

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


# profile_job = hub.submit_profile_job(
#     model=compile_job.get_target_model(),
#     device=hub.Device("QCS8550 (Proxy)"),
# )
# assert isinstance(profile_job, hub.ProfileJob)



# Path to your input PNG image
image_path = "test800.png"  # Adjust the path as needed

# Step 1: Load the image using PIL
image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already

# Step 2: Preprocess the image
# Resize to the expected input size (1600x1200) and normalize if needed
#image = image.resize((1200, 1600))  # Resize image to match the input shape (H, W)
image.show()

# Convert the image to a numpy array
image = np.array(image, dtype=np.float32)  # Convert the image to a numpy array

# Normalize the image by subtracting 0.5 (as done in the local code)
image = (image / 255.0) - 0.5  # Normalize to range [-0.5, 0.5]

# Reorder the dimensions to match (Batch, Channels, Height, Width)
image = np.transpose(image, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)

# Step 3: Submit inference job using the compiled model
inference_job = hub.submit_inference_job(
    model=compile_job.get_target_model(),
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