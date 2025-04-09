import os
import time
import logging
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision.transforms as T  # <-- Still "T"
from torchmetrics.image import PeakSignalNoiseRatio
from torch.utils.data import DataLoader, TensorDataset
import models


# Configuration
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
device = torch.device("cpu")
METHOD = "DMPHN_1_2_4"
SAMPLE_DIR = "./new_dataset/val/HAZY"
GT_DIR = "./new_dataset/val/Gt"
EXPDIR = "DMPHN_results"
os.makedirs(f"./test_results/{EXPDIR}", exist_ok=True)

# ---------------------------
# Load model components and weights
# ---------------------------

encoder_lv1 = models.Encoder().to(device)
encoder_lv2 = models.Encoder().to(device)
encoder_lv3 = models.Encoder().to(device)
decoder_lv1 = models.Decoder().to(device)
decoder_lv2 = models.Decoder().to(device)
decoder_lv3 = models.Decoder().to(device)

encoder_lv1.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv1.pkl', map_location=device))
encoder_lv2.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv2.pkl', map_location=device))
encoder_lv3.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv3.pkl', map_location=device))
decoder_lv1.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv1.pkl', map_location=device))
decoder_lv2.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv2.pkl', map_location=device))
decoder_lv3.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv3.pkl', map_location=device))

encoder_lv1.eval(), encoder_lv2.eval(), encoder_lv3.eval()
decoder_lv1.eval(), decoder_lv2.eval(), decoder_lv3.eval()

class DMPHN(nn.Module):
    def __init__(self, encoders, decoders):
        super(DMPHN, self).__init__()
        self.encoders = nn.ModuleList(encoders)  # important for NNCF
        self.decoders = nn.ModuleList(decoders)
        self._qconfig = None  # not used now

    def forward(self, x):
        if x is None:
            raise ValueError("Input tensor x is None.")
        print("Input shape:", x.shape)

        H_half = x.shape[2] // 2
        W_half = x.shape[3] // 2

        images_lv2_1 = x[:, :, :H_half, :]
        images_lv2_2 = x[:, :, H_half:, :]
        images_lv3_1 = images_lv2_1[:, :, :, :W_half]
        images_lv3_2 = images_lv2_1[:, :, :, W_half:]
        images_lv3_3 = images_lv2_2[:, :, :, :W_half]
        images_lv3_4 = images_lv2_2[:, :, :, W_half:]

        f3_1 = self.encoders[2](images_lv3_1)
        f3_2 = self.encoders[2](images_lv3_2)
        f3_3 = self.encoders[2](images_lv3_3)
        f3_4 = self.encoders[2](images_lv3_4)

        f3_top = torch.cat([f3_1, f3_2], dim=3)
        f3_bot = torch.cat([f3_3, f3_4], dim=3)
        f3 = torch.cat([f3_top, f3_bot], dim=2)

        residual_lv3_top = self.decoders[2](f3_top)
        residual_lv3_bot = self.decoders[2](f3_bot)

        f2_1 = self.encoders[1](images_lv2_1 + residual_lv3_top)
        f2_2 = self.encoders[1](images_lv2_2 + residual_lv3_bot)
        f2 = torch.cat([f2_1, f2_2], dim=2) + f3

        residual_lv2 = self.decoders[1](f2)
        f1 = self.encoders[0](x + residual_lv2) + f2
        output = self.decoders[0](f1) + 0.5

        print("Forward pass completed, output shape:", output.shape)
        return output

# Instantiate the full DMPHN model
encoders = [encoder_lv1, encoder_lv2, encoder_lv3]
decoders = [decoder_lv1, decoder_lv2, decoder_lv3]
model = DMPHN(encoders, decoders).to(device)
model.eval()

# Helper functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nonzero_parameters(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, iterations=5):
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.time()
    return (end_time - start_time) / iterations

def save_model_and_get_size(model, filename="model_pruned.pth"):
    torch.save(model.state_dict(), filename)
    return os.path.getsize(filename)

def preprocess_image(image_path):
    transform = T.Compose([
        T.CenterCrop((1072, 1920)),
        T.ToTensor(),
        T.Lambda(lambda x: x - 0.5)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(device)

def process_images(model, sample_dir, gt_dir):
    psnr_metric = PeakSignalNoiseRatio().to(device)
    total_psnr = 0.0
    count = 0
    total_time = 0.0

    for image_name in os.listdir(sample_dir):
        hazy_path = os.path.join(sample_dir, image_name)

        base, ext = os.path.splitext(image_name)
        gt_base = base.replace("_hazy", "_GT")
        gt_name = gt_base + ext
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            print(f"Missing GT image: {gt_path}")
            continue

        with torch.no_grad():
            hazy_image = preprocess_image(hazy_path)
            gt_image = preprocess_image(gt_path)
            start = time.time()
            output = model(hazy_image)
            end = time.time()

            total_time += (end - start)
            psnr_value = psnr_metric(output + 0.5, gt_image + 0.5).item()
            total_psnr += psnr_value
            count += 1
            print(f"Processed: {image_name} | PSNR: {psnr_value:.4f} | Time: {end - start:.4f}s")

    avg_psnr = total_psnr / count if count > 0 else 0
    print(f"\nAverage PSNR on dataset: {avg_psnr:.4f}")
    print(f"Total processing time for {count} images: {total_time:.4f}s")
    return avg_psnr, total_time

# Baseline evaluation
print("\n--- Baseline evaluation on dataset ---")
baseline_dataset_psnr, dataset_time = process_images(model, SAMPLE_DIR, GT_DIR)
baseline_params = count_parameters(model)
baseline_nonzero = count_nonzero_parameters(model)
baseline_inference_time = measure_inference_time(model, iterations=5)
baseline_file_size = save_model_and_get_size(model, "baseline_model.pth")

print("\nBaseline model results:")
print("  PSNR (dataset):", baseline_dataset_psnr)
print("  Number of parameters:", baseline_params)
print("  Nonzero parameters:", baseline_nonzero)
print("  Average inference time:", baseline_inference_time, "seconds per input")
print("  Model file size:", baseline_file_size, "bytes")

# Manual structured pruning (L1 norm)
threshold = 0.02
print(f"\n--- Starting manual structured pruning with L1 norm (threshold = {threshold}) ---")
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        norms = module.weight.data.abs().mean(dim=(1,2,3))
        mask = torch.ones_like(module.weight.data)
        pruned_channels = 0
        for i in range(module.weight.data.size(0)):
            if norms[i] < threshold:
                mask[i, :, :, :] = 0.0
                pruned_channels += 1
        print(f"{name}: Pruning {pruned_channels} of {module.weight.data.size(0)} channels")
        prune.custom_from_mask(module, name="weight", mask=mask)
        prune.remove(module, "weight")

print("\n--- Evaluation on dataset after manual structured pruning ---")
pruned_dataset_psnr, pruned_dataset_time = process_images(model, SAMPLE_DIR, GT_DIR)
pruned_params = count_parameters(model)
pruned_nonzero = count_nonzero_parameters(model)
pruned_inference_time = measure_inference_time(model, iterations=5)
pruned_file_size = save_model_and_get_size(model, "pruned_model.pth")

print("\nModel results after manual pruning:")
print("  PSNR (dataset):", pruned_dataset_psnr)
print("  Number of parameters:", pruned_params)
print("  Nonzero parameters:", pruned_nonzero)
print("  Average inference time:", pruned_inference_time, "seconds per input")
print("  Model file size:", pruned_file_size, "bytes")

# NNCF integration
try:
    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model, register_default_init_args
    from nncf.torch.initialization import PTInitializingDataLoader

    print("\n--- NNCF integration for filter pruning ---")
    nncf_config_dict = {
        "input_info": {"sample_size": [1, 3, 512, 512]},
        "compression": {
            "algorithm": "filter_pruning",
            "params": {
                "pruning_flops_target": 0.5,  # reduce FLOPs by 50%
                "pruning_init": "variance"
            }
        }
    }
    nncf_config = NNCFConfig(nncf_config_dict)

    dummy_data = torch.randn(100, 3, 512, 512)
    dummy_dataset = TensorDataset(dummy_data)
    train_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
    init_loader = PTInitializingDataLoader(train_loader)

    default_init_args = register_default_init_args(model, init_loader)
    if not isinstance(default_init_args, dict):
        default_init_args = {item[0]: item[1] for item in default_init_args if len(item) >= 2}
    nncf_config_dict_updated = nncf_config_dict.copy()
    nncf_config_dict_updated.update(default_init_args)
    nncf_config = NNCFConfig(nncf_config_dict_updated)

    if not hasattr(nncf_config, 'get_batch_size'):
        nncf_config.get_batch_size = lambda: 1

    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    compressed_model.eval()

    print("\n--- Evaluation of the model with NNCF compression ---")
    nncf_dataset_psnr, nncf_dataset_time = process_images(compressed_model, SAMPLE_DIR, GT_DIR)
    nncf_params = count_parameters(compressed_model)
    nncf_nonzero = count_nonzero_parameters(compressed_model)
    nncf_inference_time = measure_inference_time(compressed_model, iterations=5)
    nncf_file_size = save_model_and_get_size(compressed_model, "nncf_compressed_model.pth")

    print("\nNNCF compressed model results:")
    print("  PSNR (dataset):", nncf_dataset_psnr)
    print("  Number of parameters:", nncf_params)
    print("  Nonzero parameters:", nncf_nonzero)
    print("  Average inference time:", nncf_inference_time, "seconds per input")
    print("  Model file size:", nncf_file_size, "bytes")

except Exception as e:
    print("NNCF integration failed:", e)

# ---------------------------
# Post-Training Static Quantization
# (No fuse modules here, so it's basically un-fused static quant)
# ---------------------------
try:
    import torch.quantization

    print("\n--- Post Training Static Quantization ---")
    model_to_quantize = model.cpu().eval()

    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    qconfig = qconfig._replace(
        weight=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    )
    model_to_quantize.qconfig = qconfig

    # No fuse_modules(...) here — we skip it entirely.

    model_prepared = torch.quantization.prepare(model_to_quantize, inplace=False)

    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(1, 3, 512, 512)
            model_prepared(dummy_input)

    quantized_model = torch.quantization.convert(model_prepared, inplace=False)

    quantized_inference_time = measure_inference_time(quantized_model, iterations=5)
    quantized_file_size = save_model_and_get_size(quantized_model, "quantized_model.pth")
    quantized_dataset_psnr, _ = process_images(quantized_model, SAMPLE_DIR, GT_DIR)

    print("\nQuantized model results:")
    print("  PSNR (dataset):", quantized_dataset_psnr)
    print("  Average inference time:", quantized_inference_time, "seconds per input")
    print("  Model file size:", quantized_file_size, "bytes")

except Exception as e:
    print("Quantization failed:", e)

# -----------------------------------
# Export to ONNX
# -----------------------------------
try:
    print("\n--- Exporting to ONNX ---")
    dummy_input_onnx = torch.randn(1, 3, 512, 512, device=device)
    # When exporting, let's keep everything on CPU for safety.
    model_for_onnx = model.cpu().eval()

    torch.onnx.export(
        model_for_onnx,
        dummy_input_onnx.cpu(),
        "dmphn_no_fuse.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    print("Exported model to dmphn_no_fuse.onnx")

except Exception as e:
    print("ONNX export failed:", e)
