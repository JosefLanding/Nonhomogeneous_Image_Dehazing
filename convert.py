import torch
import models  # Import your model definitions
import os


# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
encoder_lv1 = models.Encoder().to(device)
encoder_lv2 = models.Encoder().to(device)
encoder_lv3 = models.Encoder().to(device)
decoder_lv1 = models.Decoder().to(device)
decoder_lv2 = models.Decoder().to(device)
decoder_lv3 = models.Decoder().to(device)

# Load pre-trained weights
METHOD = "DMPHN_1_2_4"
encoder_lv1.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv1.pkl', map_location=device))
encoder_lv2.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv2.pkl', map_location=device))
encoder_lv3.load_state_dict(torch.load(f'./checkpoints/{METHOD}/encoder_lv3.pkl', map_location=device))
decoder_lv1.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv1.pkl', map_location=device))
decoder_lv2.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv2.pkl', map_location=device))
decoder_lv3.load_state_dict(torch.load(f'./checkpoints/{METHOD}/decoder_lv3.pkl', map_location=device))

# Set models to evaluation mode
encoder_lv1.eval()
encoder_lv2.eval()
encoder_lv3.eval()
decoder_lv1.eval()
decoder_lv2.eval()
decoder_lv3.eval()

# Create a dummy input for ONNX conversion
dummy_input = torch.randn(1, 3, 600, 800).to(device)  # Change to the correct input size

# Define a wrapper model for ONNX export
class DMPHN_ONNX(torch.nn.Module):
    def __init__(self, encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3):
        super(DMPHN_ONNX, self).__init__()
        self.encoder_lv1 = encoder_lv1
        self.encoder_lv2 = encoder_lv2
        self.encoder_lv3 = encoder_lv3
        self.decoder_lv1 = decoder_lv1
        self.decoder_lv2 = decoder_lv2
        self.decoder_lv3 = decoder_lv3

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        H_half = H // 2
        W_half = W // 2

        images_lv2_1 = x[:, :, 0:H_half, :]
        images_lv2_2 = x[:, :, H_half:H, :]
        images_lv3_1 = images_lv2_1[:, :, :, 0:W_half]
        images_lv3_2 = images_lv2_1[:, :, :, W_half:W]
        images_lv3_3 = images_lv2_2[:, :, :, 0:W_half]
        images_lv3_4 = images_lv2_2[:, :, :, W_half:W]
        
        # images_lv2_1 = x[:, :, 0:int(H/2), :]
        # images_lv2_2 = x[:, :, int(H/2):H, :]
        # images_lv3_1 = images_lv2_1[:, :, :, 0:int(W/2)]
        # images_lv3_2 = images_lv2_1[:, :, :, int(W/2):W]
        # images_lv3_3 = images_lv2_2[:, :, :, 0:int(W/2)]
        # images_lv3_4 = images_lv2_2[:, :, :, int(W/2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)

        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3

        residual_lv2 = self.decoder_lv2(feature_lv2)
        feature_lv1 = self.encoder_lv1(x + residual_lv2) + feature_lv2
        dehazed_image = self.decoder_lv1(feature_lv1)

        
        return dehazed_image + 0.5

# Create ONNX model instance
dmphn_onnx = DMPHN_ONNX(encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3).to(device)

# Export to ONNX
onnx_filename = "dmphn_dehazing.onnx"
torch.onnx.export(
    dmphn_onnx, 
    dummy_input, 
    onnx_filename,
    export_params=True,
    opset_version=11,  # Qualcomm AI Hub supports opset 11+
    do_constant_folding=True,
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Dynamic batch size
)

print(f"ONNX model exported as {onnx_filename}")