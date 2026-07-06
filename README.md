# ai-intern-week07-edge-ai
Seventh week of the AI Engineering internship learning plan
# Week 7 - TinyML & Edge AI (LiteRT / ONNX)

This repository contains the optimization, conversion, and deployment pipeline for the Fashion-MNIST classifier on resource-constrained edge devices.

## Performance & Optimization Summary

The model was evaluated over **1,000 continuous inference calls** to monitor latency distribution, resource utilization, and potential accuracy degradation caused by fixed-point arithmetic.

| Format | Model Size (KB) | Mean Latency (ms) | p50 (ms) | p99 (ms) | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch Baseline | 405.2 KB | 4.92 ms | 4.85 ms | 9.77 ms | 88.40% |
| ONNX Runtime | 402.6 KB | 2.68 ms | 2.65 ms | 5.26 ms | 88.40% |
| LiteRT INT8 (Only) | 100.7 KB | 1.11 ms | 1.10 ms | 2.15 ms | 87.95% |

## Key Engineering Insights

1. **Storage Footprint Reduction**: Full INT8 post-training quantization successfully compressed the model by **4x** (from 402.6 KB down to 100.7 KB). This optimization directly satisfies edge deployment constraints like limited non-volatile Flash memory.
2. **Computational Speedup**: LiteRT INT8 achieved a **4.4x execution speedup** compared to the PyTorch baseline. This highlights the hardware efficiency of 8-bit integer vector operations over standard 32-bit floating-point math on CPU cores.
3. **Quantization Trade-off**: The compression and latency benefits cost an accuracy drop of only **0.45%**. This is well within acceptable production limits for a TinyML application.
4. **Tail Latency (p99)**: The p99 metric is roughly double the p50 across all formats. This reflects standard OS thread scheduling jitter and initial cache warm-up overhead during inference loops.

## How to Run

1. Generate and validate the ONNX graph:
   ```bash
   python export_onnx.py

2. Execute full integer post-training quantization:
   ```bash
    python convert_litert.py

3. Run the 1,000-cycle statistical profiling:
    ```bash
    python benchmark_script.py