import torch
import onnxruntime as ort
import time
import numpy as np
import pandas as pd

# Import structure for PyTorch
from pipeline_test.export_onnx import FlexibleNet

def benchmark_pytorch(model, input_tensor, num_runs=1000):
    latencies = []
    with torch.no_grad():
        # Warm-up run (to eliminate initial CPU loading time)
        _ = model(input_tensor)

        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to milliseconds (ms)
    return latencies

def benchmark_onnx(onnx_path, input_array, num_runs=1000):
    latencies = []
    # Initialize ONNX Runtime session on CPU
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Warm-up run
    _ = session.run(None, {input_name: input_array})

    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: input_array})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # ms
    return latencies

def calculate_metrics(latencies):
    return {
        "Mean (ms)": np.mean(latencies),
        "p50 (Median) (ms)": np.percentile(latencies, 50),
        "p99 (Tail Latency) (ms)": np.percentile(latencies, 99)
    }

def main():
    print("Starting Benchmark Edge AI (1000 iterations)...")

    # 1. Prepare test data
    torch_input = torch.randn(1, 1, 28, 28)
    numpy_input = torch_input.numpy() # ONNX uses pure NumPy as input

    # 2. Benchmark PyTorch
    pt_model = FlexibleNet(hidden_sizes=[128], activation_type="relu")
    pt_model.eval()
    pt_latencies = benchmark_pytorch(pt_model, torch_input)
    pt_metrics = calculate_metrics(pt_latencies)

    # 3. Benchmark ONNX Runtime
    onnx_path = "fashion_mnist.onnx"
    onnx_latencies = benchmark_onnx(onnx_path, numpy_input)
    onnx_metrics = calculate_metrics(onnx_latencies)

    # 4. Results shown in a table
    results = {
        "Framework": ["PyTorch (Base)", "ONNX Runtime"],
        "Mean Latency (ms)": [pt_metrics["Mean (ms)"], onnx_metrics["Mean (ms)"]],
        "p50 Latency (ms)": [pt_metrics["p50 (Median) (ms)"], onnx_metrics["p50 (Median) (ms)"]],
        "p99 Latency (ms)": [pt_metrics["p99 (Tail Latency) (ms)"], onnx_metrics["p99 (Tail Latency) (ms)"]]
    }

    df = pd.DataFrame(results)
    print("\nBenchmark Results :")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()