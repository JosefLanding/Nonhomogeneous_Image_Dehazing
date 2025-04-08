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


profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=hub.Device("QCS8550 (Proxy)"),
)
assert isinstance(profile_job, hub.ProfileJob)

