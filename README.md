# EdgeAI CNN model software & hardware co-design

### This repo contains code for a custom CNN model designed with a similar strucutre to that of VGGNet, it is quantised to INT8 on the xilinx-vitis-ai 2022 quantizer. 

### Tools used
Vitis AI, Python, Kria Pynq DPU

## Repo Structure & description

EdgeAI-CNN-KV260/
├── host/
│   ├── CNN-87acc.ipynb         # Train/test CNN (~87% accuracy)
│   ├── CNN-91acc.ipynb         # Train/test CNN (~91% accuracy)
│   ├── model-87acc.h5          # Saved Keras model (87%)
│   ├── model-91acc.h5          # Saved Keras model (91%)
│   └── Vitis-AI-Quantizer.py   # Quantization script
├── KriaKV260/
│   ├── kriaDPU-87acc.ipynb     # Run 87% model on KV260
│   ├── kriaDPU-91acc.ipynb     # Run 91% model on KV260
│   ├── model-87acc.xmodel      # Quantized model for DPU (87%)
│   ├── model-91acc.xmodel      # Quantized model for DPU (91%)
│   └── dpu/                    # DPU deployment files
│       ├── dpu-g10.bit         # FPGA bitstream
│       ├── dpu-g10.hwh         # Hardware handoff file
│       └── dpu-g10.xclbin      # DPU binary (called in Kria notebooks)
└── README.md                   # Project overview
