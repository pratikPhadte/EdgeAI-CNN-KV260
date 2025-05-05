# EdgeAI CNN Model - Software & Hardware Co-Design

This repository contains code for a custom CNN model designed with a structure similar to VGGNet, optimized and quantized to INT8 using the Xilinx Vitis AI 2022 Quantizer. The model is deployed on the Kria KV260 using the Kria Pynq DPU.

## Tools Used
- **Vitis AI**
- **Python**
- **Kria Pynq DPU**

## Video demo  
[![Watch the demo](https://img.youtube.com/vi/tzlHn4v9oog/0.jpg)](https://youtu.be/tzlHn4v9oog)


## Repo Structure & Description

EdgeAI-CNN-KV260/<br>
├── host/<br>
│ ├── CNN-87acc.ipynb # Train/test CNN (~87% accuracy)<br>
│ ├── CNN-91acc.ipynb # Train/test CNN (~91% accuracy)<br>
│ ├── model-87acc.h5 # Saved Keras model (87%)<br>
│ ├── model-91acc.h5 # Saved Keras model (91%)<br>
│ └── Vitis-AI-Quantizer.py # Quantization script<br>
├── KriaKV260/<br>
│ ├── kriaDPU-87acc.ipynb # Run 87% model on KV260<br>
│ ├── kriaDPU-91acc.ipynb # Run 91% model on KV260<br>
│ ├── model-87acc.xmodel # Quantized model for DPU (87%)<br>
│ ├── model-91acc.xmodel # Quantized model for DPU (91%)<br>
│ └── dpu/ # DPU deployment files<br>
│ ├── dpu-g10.bit # FPGA bitstream<br>
│ ├── dpu-g10.hwh # Hardware handoff file<br>
│ └── dpu-g10.xclbin # DPU binary (used in Kria notebooks)<br>
└── README.md # Project overview<br>

## Model Overview

The CNN model is inspired by the **VGGNet architecture**, optimized for edge deployment. It consists of several convolutional layers with batch normalization and max-pooling layers, followed by dense layers for classification. The model was quantized to INT8 using the Vitis AI Quantizer for efficient deployment on Xilinx devices.

> **Total Parameters:** `370,402`  <br>
> **Trainable Parameters:** `369,602`  <br>
> **Non-trainable Parameters:** `800`  <br>

---

###  Layer-wise Breakdown<br>

| Layer Type            | Output Shape        | Parameters |<br>
|-----------------------|---------------------|------------|<br>
| **Input**             | (32, 32, 3)         | –          |<br>
| Conv2D (32 filters)   | (32, 32, 32)        | 896        |<br>
| BatchNormalization    | (32, 32, 32)        | 128        |<br>
| Conv2D (32 filters)   | (32, 32, 32)        | 9,248      |<br>
| BatchNormalization    | (32, 32, 32)        | 128        |<br>
| MaxPooling2D          | (16, 16, 32)        | 0          |<br>
| Conv2D (32 filters)   | (16, 16, 32)        | 9,248      |<br>
| BatchNormalization    | (16, 16, 32)        | 128        |<br>
| Conv2D (64 filters)   | (16, 16, 64)        | 18,496     |<br>
| BatchNormalization    | (16, 16, 64)        | 256        |<br>
| MaxPooling2D          | (8, 8, 64)          | 0          |<br>
| Conv2D (120 filters)  | (8, 8, 120)         | 69,240     |<br>
| BatchNormalization    | (8, 8, 120)         | 480        |<br>
| Conv2D (120 filters)  | (8, 8, 120)         | 129,720    |<br>
| BatchNormalization    | (8, 8, 120)         | 480        |<br>
| MaxPooling2D          | (4, 4, 120)         | 0          |<br>
| Flatten               | (1920,)             | 0          |<br>
| Dense (64 units)      | (64,)               | 122,944    |<br>
| Dense (120 units)     | (120,)              | 7,800      |<br>
| Dense (10 units)      | (10,)               | 1,210      |<br>


## Getting Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/pratikPhadte/EdgeAI-CNN-KV260.git
    ```

2. **Host-side setup**:
    - Use the provided Jupyter notebooks to train and test the CNN models.
    - Quantize the model using `Vitis-AI-Quantizer.py`.

3. **Kria KV260 setup**:
    - Follow the instructions in the `kriaDPU-*.ipynb` notebooks to deploy the quantized models on the Kria KV260 using the Pynq DPU.
    - The DPU bitstream and hardware files are provided in the `dpu/` directory.

