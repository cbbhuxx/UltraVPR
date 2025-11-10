# =====================================================================================
# Preamble: Import necessary libraries
# =====================================================================================
import utils
from main import parser
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Import libraries for ONNX export and verification
import onnx
import onnxruntime
import onnxsim  # Import the onnx-simplifier library

# =====================================================================================
# Section 1: Wrapper Classes for the Exported Model
#
# Why are these needed? The .export() method from the e2cnn library converts
# custom equivariant layers into standard PyTorch layers, but it returns them inside
# a `ModuleDict`, which is just a container and has no `forward()` method.
# These wrapper classes reconstruct the model's execution flow (the `forward` method)
# using the standard layers contained in the exported `ModuleDict`.
# =====================================================================================

# Wrapper for a Bottleneck block
class ExportedBottleneck(nn.Module):
    def __init__(self, block_dict):
        super().__init__()
        self.layers = nn.ModuleDict(block_dict)
        self.has_downsample = 'downsample' in self.layers

    def forward(self, x):
        identity = x
        # Mimics the forward pass of a standard ResNet Bottleneck block
        out = self.layers['conv1'](x)
        out = self.layers['bn1'](out)
        out = self.layers['relu1'](out)
        out = self.layers['conv2'](out)
        out = self.layers['bn2'](out)
        out = self.layers['relu2'](out)
        out = self.layers['conv3'](out)
        out = self.layers['bn3'](out)
        if self.has_downsample:
            identity = self.layers['downsample'](x)
        out += identity # The core residual connection
        out = self.layers['relu3'](out)
        return out

# Wrapper for a Basic block
class ExportedBasicBlock(nn.Module):
    def __init__(self, block_dict):
        super().__init__()
        self.layers = nn.ModuleDict(block_dict)
        self.has_downsample = 'downsample' in self.layers
    
    def forward(self, x):
        identity = x
        # Mimics the forward pass of a standard ResNet Basic block
        out = self.layers['conv1'](x)
        out = self.layers['bn1'](out) 
        out = self.layers['relu1'](out)
        out = self.layers['conv2'](out)
        out = self.layers['bn2'](out)
        if self.has_downsample:
            identity = self.layers['downsample'](x)
        out += identity
        out = self.layers['relu2'](out)
        return out

# Wrapper for a ResLayer, which contains multiple blocks
class ExportedResLayer(nn.Module):
    def __init__(self, layer_dict):
        super().__init__()
        blocks = OrderedDict()
        # Iterate through blocks ('0', '1', '2', ...) in sorted order
        for block_name in sorted(layer_dict.keys(), key=int):
            block_dict = layer_dict[block_name]
            # Intelligently choose the correct wrapper based on the presence of 'conv3'
            if 'conv3' in block_dict:
                block = ExportedBottleneck(block_dict)
            else:
                block = ExportedBasicBlock(block_dict)
            blocks[block_name] = block
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, x):
        # Pass the input sequentially through each block in the layer
        for block_name in self.blocks:
            x = self.blocks[block_name](x)
        return x

# The main wrapper for the entire E2ResNet backbone
class ExportedE2ResNet(nn.Module):
    def __init__(self, exported_dict):
        super().__init__()
        layers = OrderedDict()
        for name, module_dict in exported_dict.items():
            # If the module is a 'layer' (e.g., 'layer1'), wrap it with ExportedResLayer
            if name.startswith('layer'):
                layers[name] = ExportedResLayer(module_dict)
            else: # Otherwise, it's a simple module (Conv2d, ReLU, etc.)
                layers[name] = module_dict
        self.layers = nn.ModuleDict(layers)
        
        # These attributes must match the original E2ResNet configuration
        self.res_layers_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self.out_indices = (3,) 

    def forward(self, x):
        # Reconstruct the forward pass of the original E2ResNet
        x = self.layers['conv1'](x)
        x = self.layers['bn1'](x)
        x = self.layers['relu'](x)
        x = self.layers['maxpool'](x)
        outs = []
        for i, layer_name in enumerate(self.res_layers_names):
            res_layer = self.layers[layer_name]
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        out = self.layers['gpool'](outs[0]) 
        return out

# =====================================================================================
# Section 2: Model Loading and Conversion Function
# =====================================================================================

def load_and_prepare_model(ckpt_path):
    # Load model configuration and create the initial e2cnn model
    opt = parser.parse_args()
    model = utils.set_model(opt)
    
    # Load pretrained weights
    if ckpt_path != "":
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'], strict=False)
        print(f"Loaded model from {ckpt_path} Successfully!")
    
    model.eval()
    
    # The core conversion step: call .export() on the e2cnn backbone
    backbone_to_export = model.backbone
    print("Exporting the E2ResNet backbone to standard PyTorch modules...")
    exported_dict = backbone_to_export.export()
    
    # --- Debugging Helper ---
    # Print the keys to help verify the layer names used in the wrapper classes
    print("Available layers in exported backbone:", exported_dict.keys())
    if 'layer1' in exported_dict and '0' in exported_dict['layer1']:
        print("Keys inside a sample Bottleneck block (layer1['0']):", exported_dict['layer1']['0'].keys())
    # ---
    
    print("Backbone exported to ModuleDict.")
    
    # Wrap the exported ModuleDict with our custom class to restore the forward pass
    exported_backbone = ExportedE2ResNet(exported_dict)
    print("Created new backbone with a forward method, ready for acceleration.")
    
    # Replace the original backbone with the new, exported, acceleration-friendly backbone
    model.backbone = exported_backbone
    return model

# Final model class that combines backbone and aggregator
class UltraVPR(nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

# Helper function to convert a PyTorch tensor to a NumPy array for ONNX Runtime
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# =====================================================================================
# Section 3: Main Execution Block
# =====================================================================================
if __name__ == "__main__":
    # --- 1. Setup Environment and Paths ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_ckpt_path = '/media/cbbhuxx/UD210/my-paper-code/e2resnet50_c8_se2gem_32/checkpoints/model_best.pth.tar'
    onnx_export_path = "./ultravpr_original.onnx"
    onnx_simplified_export_path = "./ultravpr_simplified.onnx"

    # --- 2. Load and Prepare the PyTorch Model ---
    # This function handles loading weights and converting the backbone
    prepared_model = load_and_prepare_model(model_ckpt_path)
    pytorch_model = UltraVPR(prepared_model.backbone, prepared_model.aggregator).eval().to(device)

    # --- 3. Create Dummy Input Data ---
    # Use a fixed seed for reproducibility, ensuring the input is the same every time
    torch.manual_seed(0)
    dummy_image = torch.randn(1, 3, 300, 400, requires_grad=False).to(device)

    # --- 4. Run Inference with the PyTorch Model ---
    print("\n--- Running PyTorch model ---")
    pytorch_output = pytorch_model(dummy_image)
    print("PyTorch model output shape:", pytorch_output.shape)

    # --- 5. Export the Model to ONNX ---
    print(f"\n--- Exporting model to ONNX at {onnx_export_path} ---")
    with torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            dummy_image,
            onnx_export_path,
            verbose=False,
            input_names=["image"],
            output_names=["output"],
            opset_version=17,
        )
    print("ONNX export complete!")
    
    # --- 6. Simplify the Exported ONNX Model ---
    print(f"\n--- Simplifying ONNX model ---")
    try:
        onnx_model = onnx.load(onnx_export_path)
        model_simplified, check = onnxsim.simplify(onnx_model)
        assert check, "ONNX model simplification check failed."
        onnx.save(model_simplified, onnx_simplified_export_path)
        print(f"Simplified model saved to {onnx_simplified_export_path}")
    except Exception as e:
        print(f"\n❌ FAILED: An error occurred during ONNX simplification: {e}")
        exit()

    # --- 7. Verify the SIMPLIFIED ONNX Model ---
    print(f"\n--- Verifying simplified ONNX model: {onnx_simplified_export_path} ---")
    try:
        # Check that the simplified model is structurally valid
        onnx_model_check = onnx.load(onnx_simplified_export_path)
        onnx.checker.check_model(onnx_model_check)
        print("ONNX model structure is valid.")

        # Create an ONNX Runtime inference session
        ort_session = onnxruntime.InferenceSession(onnx_simplified_export_path, providers=['CPUExecutionProvider'])

        # Prepare inputs for ONNX Runtime (must be NumPy arrays)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_image)}
        
        # Run inference with ONNX Runtime
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outputs[0]
        
        print("ONNX model output shape:", onnx_output.shape)

        # Compare the outputs of PyTorch and ONNX Runtime
        # This will raise an AssertionError if the outputs are not close enough
        np.testing.assert_allclose(to_numpy(pytorch_output), onnx_output, rtol=1e-02, atol=1e-03)
        
        print("\n✅ SUCCESS: The simplified ONNX model's output matches the PyTorch model's output!")

    except Exception as e:
        print(f"\n❌ FAILED: An error occurred during ONNX verification: {e}")

    print("\nDone.")