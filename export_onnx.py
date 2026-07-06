import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import os


# Authoring the FlexibleNet model from Week 3

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

    def forward(self, x: torch.Tensor):
        # Flatten image from (Batch, 1, 28, 28) to (Batch, 784)
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
    print("Starting PyTorch to ONNX implementation...")

    # Instantiate Week 3 configuration
    torch_model = FlexibleNet(hidden_sizes=[128], activation_type="relu")

    # Loading trained weights
    weights_path = "model.pt"
    if os.path.exists(weights_path):
        print(f"Loading pre-trained weights from {weights_path}...")
        torch_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        print("Weights not found at path; proceeding with initialized weights.")

    # Put the model in evaluation mode
    torch_model.eval()

    # Export the model to ONNX format using Dynamo
    # Creating a single-batch input resembling Fashion-MNIST data shape
    example_inputs = (torch.randn(1, 1, 28, 28),)

    print("Exporting the PyTorch model via Dynamo Graph Capture...")
    onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)

    # Save and verify the ONNX model file
    onnx_filename = "fashion_mnist.onnx"
    onnx_program.save(onnx_filename)
    print(f"ONNX model program saved successfully as '{onnx_filename}'")

    # Load back and validate integrity
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("Validated by ONNX checker library!")

    # Execute the ONNX model with ONNX Runtime
    print("\nExecuting inference via ONNX Runtime Engine...")

    # Convert PyTorch tensors to standard CPU Numpy matrices
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]

    ort_session = onnxruntime.InferenceSession(
        onnx_filename,
        providers=["CPUExecutionProvider"]
    )

    # Programmatically extract the input names from the generated graph
    onnxruntime_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
    }

    # Execute graph calculation
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    # Compare PyTorch results against ONNX Runtime (Numerical Validation)
    print("\nQuantifying output discrepancies (PyTorch vs ONNX Runtime)...")
    with torch.no_grad():
        torch_outputs = torch_model(*example_inputs)

    # Convert ONNX outputs back to PyTorch structures for mathematical assertion
    torch.testing.assert_close(torch_outputs, torch.tensor(onnxruntime_outputs))
    print("Success! PyTorch and ONNX Runtime outputs match perfectly to high precision.")
    print(f"Sample raw prediction array (logits):\n {onnxruntime_outputs}")

    # Define the official Fashion-MNIST class names mapping
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Find the index of the highest logit score (Prediction)
    predicted_index = onnxruntime_outputs[0].argmax()
    predicted_label = classes[predicted_index]

    print(f"\nONNX Runtime Prediction Interpretation:")
    print(f"Highest Score Index : {predicted_index}")
    print(f"Predicted Object    : '{predicted_label}'")

if __name__ == "__main__":
    main()