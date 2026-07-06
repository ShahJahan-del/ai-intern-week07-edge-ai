# ai-intern-week07-edge-ai
Seventh week of the AI Engineering internship learning plan

# AI Intern - Week 07: TinyML & Edge AI (Model Optimization, ONNX & LiteRT)

This repository contains the deliverables for Week 7 of the AI Internship, focusing on Edge AI and TinyML optimization techniques. The goal is to optimize a trained model for resource-constrained hardware (e.g., embedded devices, microcontrollers) using ONNX Runtime and LiteRT (formerly TensorFlow Lite) post-training quantization.

## Performance & Optimization Benchmarks

The following table summarizes the file sizes and inference latencies evaluated across **1,000 continuous inference calls** (using a single-batch setup typical of real-time edge processing).

| Format | Size (KB) | Mean Latency | p50 (Median) | p99 (Tail) | Accuracy Drop |
|:---|:---|:---|:---|:---|:---|
| **PyTorch (Base)** | 399.76 KB | 0.03956 ms | 0.03830 ms | 0.05010 ms | 0.00% (Ref) |
| **ONNX Runtime** | 1.25 KB | 0.02016 ms | 0.01970 ms | 0.02911 ms | 0.00% |
| **LiteRT (Float32)** | 1.25 KB | 0.02116 ms | 0.02069 ms | 0.03057 ms | 0.00% |
| **LiteRT (INT8 Quantized)**| 0.31 KB | 0.01915 ms | 0.01872 ms | 0.02765 ms | < 0.50% |

---

## Embedded Systems Insights

### 1. Architectural Size Compression
* **The Issue:** Native deep learning frameworks (like PyTorch) generate large weight binaries (`.pt`) holding runtime metadata, which exceeds the strict static Flash/SRAM limitations of edge devices.
* **The Solution:** By stripping non-essential training graphs and mapping operations to an inference-only framework (ONNX/LiteRT), size drops dramatically to **1.25 KB**.
* **INT8 Post-Training Quantization:** Converting weights from Float32 (32-bit floating-point) to INT8 (8-bit signed integer) yields a strict **4x storage reduction** (down to **0.31 KB**), saving precious non-volatile memory space on embedded chips.

### 2. Runtime Speed & Determinism
* **ONNX Runtime Engine:** On local CPU computing, ONNX Runtime optimizes the underlying mathematical execution graph (e.g., operator fusion), dividing the mean execution latency by **2x** compared to base PyTorch.
* **Tail Latency (p99):** In real-time embedded safety systems, worst-case execution time (WCET) matters more than the average speed. Base PyTorch suffers from latency spikes (**0.050 ms**), whereas optimized TinyML runtimes enforce a highly predictable execution ceiling below **0.030 ms**.

---

## Repository Structure

```text
ai-intern-week07-edge-ai/
├── export_onnx.py         # Script converting the Week 3 PyTorch model into an ONNX graph
├── convert_litert.py      # Optimization script detailing INT8 quantization logic
├── benchmark_script.py    # Unified benchmarking harness (1,000 iterations tracking p50/p99)
├── fashion_mnist.onnx     # Optimized ONNX model file (1.25 KB)
├── model_standard.tflite  # LiteRT Float32 deployment model
├── model_quantized.tflite # TinyML INT8 highly compressed model (0.31 KB)
└── README.md              # Project report and metrics overview (This file)

Reprodubility Instructions
1. Installation
Install the necessary runtime libraries:

Bash
pip install torch torchvision onnx onnxruntime pandas tabulate
2. Run the Pipelines
Generate the optimized graphs and launch the physical benchmark test suite:

Bash
python export_onnx.py
python convert_litert.py
python benchmark_script.py