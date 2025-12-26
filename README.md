<div align="center">
<h1>
  SAHI Zero-Copy: GPU-Optimized Slicing Aided Hyper Inference
</h1>

<h4>
  A fork of SAHI with GPU tensor-based inference and batch tile processing for maximum performance
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sahi-sliced-inference-overview.avif">
</h4>

</div>

## 🚀 Performance Comparison

| Method | Time (2986x1680 image) | Speedup |
|--------|------------------------|---------|
| Original SAHI (CPU) | 2.83s | 1.00x |
| GPU Sequential | 0.55s | 5.14x |
| **GPU Batch** | **0.18s** | **16.17x** |

## ✨ What's New in This Fork?

### 1. GPU Tensor-Based Operations
All image processing stays on GPU memory, minimizing CPU-GPU data transfers.

### 2. Batch Tile Inference
Process all tiles in a **single forward pass** instead of one-by-one.

### 3. New Functions

| Function | Description |
|----------|-------------|
| `slice_image_gpu()` | Slice images directly as GPU tensors |
| `get_prediction_gpu()` | Run prediction on GPU tensor input |
| `get_sliced_prediction_gpu()` | Full sliced inference pipeline on GPU |
| `get_prediction_gpu_batch()` | Batch inference for multiple images |
| `perform_inference_gpu()` | Direct GPU tensor inference in models |
| `read_image_as_tensor()` | Load images as PyTorch tensors |
| `SliceImageResult.get_batch_tiles()` | Stack all tiles into batch tensor |

---

## 📦 Installation

```bash
# Clone this fork
git clone https://github.com/your-repo/sahi-zero-copy.git
cd sahi-zero-copy

# Install dependencies
pip install -e .

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics
pip install ultralytics
```

---

## 🔥 Quick Start

### Basic GPU Inference

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction_gpu
from sahi.utils.cv import read_image_as_tensor

# Load model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    device="cuda",
)

# Read image as GPU tensor
image_tensor = read_image_as_tensor("image.jpg", device="cuda")

# Run GPU sliced prediction
result = get_sliced_prediction_gpu(
    image=image_tensor,
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    device="cuda",
)

print(f"Detected {len(result.object_prediction_list)} objects")
```

### 🚀 Batch Tile Inference (Fastest)

```python
# Enable batch inference - all tiles processed in ONE forward pass
result = get_sliced_prediction_gpu(
    image=image_tensor,
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    batch_tile=True,  # ← Enable batch inference
    device="cuda",
)
```

### Sequential vs Batch Comparison

```python
import time

# Sequential (default)
start = time.time()
result_seq = get_sliced_prediction_gpu(
    image=image_tensor,
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    batch_tile=False,
    device="cuda",
)
print(f"Sequential: {time.time() - start:.3f}s")

# Batch (faster)
start = time.time()
result_batch = get_sliced_prediction_gpu(
    image=image_tensor,
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    batch_tile=True,
    device="cuda",
)
print(f"Batch: {time.time() - start:.3f}s")
```

---

## 📖 API Reference

### `get_sliced_prediction_gpu()`

Main function for GPU-based sliced inference.

```python
def get_sliced_prediction_gpu(
    image: str | torch.Tensor,           # Image path or GPU tensor
    detection_model,                      # Detection model
    slice_height: int = None,             # Height of each slice
    slice_width: int = None,              # Width of each slice
    overlap_height_ratio: float = 0.2,    # Vertical overlap ratio
    overlap_width_ratio: float = 0.2,     # Horizontal overlap ratio
    perform_standard_pred: bool = True,   # Also run on full image
    postprocess_type: str = "GREEDYNMM",  # NMM, GREEDYNMM, NMS, LSNMS
    postprocess_match_metric: str = "IOS", # IOU or IOS
    postprocess_match_threshold: float = 0.5,
    device: str = "cuda",                 # Device to use
    batch_tile: bool = False,             # Enable batch inference
    verbose: int = 1,
) -> PredictionResult:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | str \| torch.Tensor | - | Image path or GPU tensor (CHW format) |
| `detection_model` | DetectionModel | - | Loaded detection model |
| `slice_height` | int | None | Height of each slice (auto if None) |
| `slice_width` | int | None | Width of each slice (auto if None) |
| `overlap_height_ratio` | float | 0.2 | Vertical overlap between slices |
| `overlap_width_ratio` | float | 0.2 | Horizontal overlap between slices |
| `perform_standard_pred` | bool | True | Run prediction on full image too |
| `postprocess_type` | str | "GREEDYNMM" | Postprocessing algorithm |
| `postprocess_match_metric` | str | "IOS" | IOU or IOS for matching |
| `postprocess_match_threshold` | float | 0.5 | Threshold for merging |
| `device` | str | "cuda" | Device for inference |
| **`batch_tile`** | **bool** | **False** | **Enable batch tile inference** |
| `verbose` | int | 1 | Verbosity level (0, 1, 2) |

### `read_image_as_tensor()`

Load image directly as GPU tensor.

```python
from sahi.utils.cv import read_image_as_tensor

# From file path
tensor = read_image_as_tensor("image.jpg", device="cuda")
# Shape: (3, H, W), dtype: uint8

# From numpy array
import numpy as np
img_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
tensor = read_image_as_tensor(img_np, device="cuda")

# From PIL Image
from PIL import Image
pil_img = Image.open("image.jpg")
tensor = read_image_as_tensor(pil_img, device="cuda")
```

### `slice_image_gpu()`

Slice images on GPU.

```python
from sahi.slicing import slice_image_gpu

slice_result = slice_image_gpu(
    image=image_tensor,       # GPU tensor or path
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    device="cuda",
)

# Access slices
print(f"Number of tiles: {len(slice_result)}")
print(f"Starting pixels: {slice_result.starting_pixels}")

# Stack all tiles into batch
batch_tensor = slice_result.get_batch_tiles()  # (N, C, H, W)
```

---

## ⚙️ Postprocessing Methods

SAHI supports different algorithms for merging overlapping predictions:

| Method | Description | Speed |
|--------|-------------|-------|
| `GREEDYNMM` | Greedy Non-Maximum Merging (default) | Medium |
| `NMM` | Non-Maximum Merging | Fast |
| `NMS` | Non-Maximum Suppression | Fast |
| `LSNMS` | Large-Scale NMS (requires `pip install lsnms`) | Fast |

```python
# Use different postprocessing
result = get_sliced_prediction_gpu(
    image=image_tensor,
    detection_model=detection_model,
    postprocess_type="NMS",           # Change algorithm
    postprocess_match_metric="IOU",   # IOU or IOS
    postprocess_match_threshold=0.5,  # Merge threshold
    batch_tile=True,
)
```

---

## 🧪 Benchmarking

### Run Benchmark Script

```bash
# Compare all 3 methods
python benchmark_sahi_comparison.py

# With custom image
python benchmark_sahi_comparison.py --image path/to/image.jpg

# With larger test image
python benchmark_sahi_comparison.py --image-height 4096 --image-width 4096

# Skip original SAHI (GPU only)
python benchmark_sahi_comparison.py --skip-original
```

### Test with Visualizations

```bash
# Test on terrain3.png with output images
python test_sahi_methods_visual.py

# Results saved to benchmark_results/
# - result_original_sahi.png/json
# - result_gpu_sequential.png/json
# - result_gpu_batch.png/json
```

### Compare Postprocessing Methods

```bash
python test_postprocessing_methods.py

# Results saved to postprocess_results/
```

---

## 📊 Memory Considerations

### Batch Tile Memory Usage

When `batch_tile=True`, all tiles are loaded into GPU memory at once:

| Image Size | Tiles | GPU Memory (approx) |
|------------|-------|---------------------|
| 2048×2048 | 9 | ~350 MB |
| 4096×4096 | 100 | ~2 GB |
| 8192×8192 | 400+ | ~8 GB |

> ⚠️ For very large images, use `batch_tile=False` to avoid OOM errors.

### Memory Optimization Tips

```python
import torch

# Clear cache before inference
torch.cuda.empty_cache()

# For large images, use sequential mode
result = get_sliced_prediction_gpu(
    image=large_image,
    batch_tile=False,  # Sequential to save memory
)

# Monitor memory
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

---

## 🔧 Advanced Usage

### Custom Detection Model

```python
from sahi.models.base import DetectionModel

class MyCustomModel(DetectionModel):
    def load_model(self):
        # Load your model
        self.model = ...
    
    def perform_inference_gpu(self, image: torch.Tensor):
        # Implement GPU inference
        predictions = self.model(image)
        self._original_predictions = predictions
```

### Video Inference

```python
import cv2
from sahi.predict import get_sliced_prediction_gpu
from sahi.utils.cv import read_image_as_tensor

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = read_image_as_tensor(frame_rgb, device="cuda")
    
    # Fast inference with batch
    result = get_sliced_prediction_gpu(
        image=tensor,
        detection_model=detection_model,
        batch_tile=True,
    )
    
    # Process results...
```

---

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA 11.6+ (recommended)
- Ultralytics 8.0+

---
