# Installation guide for RIPT-VLA

### To run RIPT-VLA + QueST, you only need to install steps 0, 1, and 2.

### To run RIPT-VLA + OpenVLA-OFT, you will additionally need to install steps 3.

## 0. Base RIPT-VLA environment
### Create and activate conda environment
```
conda create -n ript_vla python=3.10.14
conda activate ript_vla
```

### Install PyTorch
```
python -m pip install torch==2.2.0 torchvision==0.17.0
```

## 1. QueST

### Install dependency (mostly inherited from QueST)
```
git clone https://github.com/Ariostgx/ript-vla.git
cd ript-vla
python -m pip install -e .
```

The codebase and dependency is based on the [QueST](https://quest-model.github.io/).

## 2. LIBERO
### Library
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```

### Dataset
Please download the libero data following official [docs](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html#datasets).

## 3. OpenVLA-OFT

### Create and activate a new conda environment for openvla-oft + RIPT-VLA
- We recommend creating a new conda environment for openvla-oft + RIPT-VLA to avoid dependency conflicts with QueST.

```
conda create -n ript_vla_openvla_oft python=3.10 -y
conda activate ript_vla_openvla_oft
pip3 install torch torchvision torchaudio
```

### Clone openvla-oft repo and pip install to download dependencies
```
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .
```

### Install accleration libraries
```
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.7.4.post1" --no-build-isolation
pip install accelerate==1.6.0
```

### Install RIPT-VLA dependency
```
cd ript-vla
python -m pip install -e .
```

The codebase and dependency is based on the [QueST](https://quest-model.github.io/).

### Install LIBERO
```
cd LIBERO
python -m pip install -e .
```

The codebase and dependency is based on the [OpenVLA-OFT](https://github.com/moojink/openvla-oft).
