# Lean Suppression Fusion Enhancement Network for Multispectral Object Detection

Official implementation of **LSFENet (Lean Suppression Fusion Enhancement Network)** — an improved **RGB-T (Visible + Thermal) multispectral object detection framework** built on [Ultralytics YOLOv8/11](https://github.com/ultralytics/ultralytics) (v8.3.75). LSFENet adopts a **dual-stream backbone with mid-fusion** strategy, and introduces lightweight cross-modal feature fusion and enhancement modules to effectively exploit the complementarity between visible and infrared modalities, improving detection performance in low-light and nighttime scenarios.

## ✨ Highlights

- **Dual-stream backbone**: RGB and infrared features are extracted by two independent streams (the 6-channel input is split via `SilenceChannel`), then fused at intermediate layers;
- **IFSF module** (Infrared-Feature Selection Fusion): a global + local dual-branch channel attention fusion module for adaptive RGB / IR feature fusion in the backbone;
- **SFFE_Suppress module**: a spatial feature fusion / suppression module that further refines the fused features along the top-down path in the neck;
- **DySample upsampling**: replaces nearest-neighbor upsampling with dynamic point sampling;
- **Multiple fusion variants**: model configs with mid-fusion at different stages, `yolo11-RGBT-midfusion-P0~P5-LSFENet.yaml`;
- **Fully compatible with the original Ultralytics API**: training, validation, inference and export all work as usual.

## 📁 Project Structure

```
LSFENet/
├── train.py                     # Training entry point
├── val.py                       # Validation entry point
├── get_FPS.py                   # Inference speed (FPS) benchmark
├── heatmap_RGBT.py              # Heatmap visualization
├── requirements.txt             # Core dependencies
├── environment.yml              # Full conda environment (exact reproduction)
└── ultralytics/                 # Modified Ultralytics framework
    ├── cfg/models/v11-RGBT/     # ★ LSFENet and RGBT fusion model configs
    ├── cfg/models/v5-RGBT/      # YOLOv5 RGBT baseline configs
    ├── cfg/models/v8-RGBT/      # YOLOv8 RGBT baseline configs
    ├── cfg/datasets/            # Dataset configs (LLVIP, M3FD, FLIR, etc.)
    └── nn/modules/              # ★ Custom modules
        ├── attention.py         #   IFSF / SFFE_Suppress
        ├── dysample.py          #   DySample
        └── conv.py              #   SilenceChannel, etc.
```

## 🔧 Installation

The original experiments were conducted with **Python 3.8 + PyTorch 1.12.1 + CUDA 10.2** (Linux).

**Option A — reproduce the exact environment (recommended):**

```bash
conda env create -f environment.yml
conda activate LSFENet
```

**Option B — manual install:**

```bash
# 1. Create the environment
conda create -n lsfenet python=3.8 -y
conda activate lsfenet

# 2. Install PyTorch matching your CUDA driver (see https://pytorch.org/get-started/locally/)
pip install torch==1.12.1 torchvision==0.13.1

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation

Supported datasets: [LLVIP](https://github.com/xiaoma-L/LLVIP), [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), [FLIR](https://www.flir.com/oem/adas/adas-dataset-form/).

Datasets should be organized in YOLO format, where each visible image has an infrared counterpart with the **same filename and resolution**:

```
datasets/LLVIP/
├── visible/          # Visible images
│   ├── train/
│   ├── test/
│   ├── train.txt     # Image path lists
│   └── test.txt
└── infrared/         # Infrared images (same structure as above)
```

Then edit the corresponding config file under `ultralytics/cfg/datasets/` (e.g. `LLVIP.yaml`, `M3FD.yaml`, `FLIR.yaml`) and update the paths:

```yaml
train: /your/path/LLVIP/visible/train.txt
val:   /your/path/LLVIP/visible/test.txt
nc: 1
names: [ 'person' ]
```

## 🚀 Quick Start

### Training

Edit `train.py` to select the model config and dataset, then run:

```bash
python train.py
```

```python
model = YOLO('./ultralytics/cfg/models/v11-RGBT/yolo11-RGBT-midfusion-P3-LSFENet.yaml')
model.train(data='./ultralytics/cfg/datasets/LLVIP.yaml',
            imgsz=640,
            epochs=300,
            batch=16,
            use_simotm="RGBRGB6C",  # Concatenate RGB + IR into a 6-channel input
            channels=6,
            device='0',
            )
```

- `use_simotm="RGBRGB6C"`: automatically concatenates the visible and infrared images into a 6-channel input during training;
- The `scales` section of each model config supports five model sizes (`n / s / m / l / x`); e.g. `yolo11-RGBT-midfusion-P3-LSFENet.yaml` is built at the `m` scale by default.

### Validation

```bash
python val.py
```

Update the weight path in `val.py`, and set `save_json=True` for COCO metrics.

### FPS Benchmark

```bash
python get_FPS.py --weights runs/LLVIP/xxx/weights/best.pt --imgs 640 640 --device 0
```

### Heatmap Visualization

```bash
python heatmap_RGBT.py
```

## 🙏 Acknowledgements

This project is built upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and benefits from many open-source RGBT detection works in the community.

## 📜 License

This project follows the original [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) license. If this project helps your research, a ⭐ and a citation are appreciated.
