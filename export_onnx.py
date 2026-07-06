import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 1. Week 3 architecture copy
class FlexibleNet(nn.Module):
    def __init__(self, hidden_sizes, activation_type="relu"):
        super().__init__()
        self.activation_type = activation_type.lower()

        layers = []
        input_dim = 784
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            input_dim = hidden_size

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(input_dim, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation_type == "relu":
                x = F.relu(x)
            elif self.activation_type == "sigmoid":
                x = torch.sigmoid(x)
        x = self.output_layer(x)
        return x

def main():
    # 2. Instanciate configuration of "model_1" (hidden_sizes=[128])
    model = FlexibleNet(hidden_sizes=[128], activation_type="relu")

    # Load trained model weights
    model_path = "model.pt"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} not found. Make sure the file is in the same folder.")
        return

    model.eval()

    # 3. Dummy input for Fashion-MNIST (Batch size = 1, 1 canal, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 4. Export ONNX
    onnx_path = "fashion_mnist.onnx"
    print(f"Exporting model to {onnx_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    if os.path.exists(onnx_path):
        print(f"Export successful ! File size : {os.path.getsize(onnx_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()