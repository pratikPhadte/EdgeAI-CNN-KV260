# EdgeAI CNN Model - Software & Hardware Co-Design

This repository contains code for a custom CNN model designed with a structure similar to VGGNet, optimized and quantized to INT8 using the Xilinx Vitis AI 2022 Quantizer. The model is deployed on the Kria KV260 using the Kria Pynq DPU.

## Tools Used
- **Vitis AI**
- **Python**
- **Kria Pynq DPU**

## Video demo  
[here](https://github.com/pratikPhadte/EdgeAI-CNN-KV260/blob/main/demo-g-10.mp4).<br>

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

