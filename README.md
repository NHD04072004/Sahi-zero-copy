<div align="center">
<h1>
  SAHI Zero-Copy: GPU-Optimized Slicing Aided Hyper Inference
</h1>

<h4>
  A fork of SAHI with GPU tensor-based inference for zero-copy processing
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sahi-sliced-inference-overview.avif">
</h4>

> **🚀 This fork adds GPU tensor-based operations to minimize CPU-GPU data transfers**

</div>

## What's Different in This Fork?

This fork extends the original SAHI library with **GPU-optimized, tensor-based inference** to achieve near zero-copy processing:

### ✅ New GPU Functions

| Function | Description |
|----------|-------------|
| `slice_image_gpu()` | Slice images directly as GPU tensors |
| `get_prediction_gpu()` | Run prediction on GPU tensor input |
| `get_sliced_prediction_gpu()` | Full sliced inference pipeline on GPU |
| `perform_inference_gpu()` | Direct GPU tensor inference in models |
| `read_image_as_tensor()` | Load images as PyTorch tensors |

### ⚡ Performance Optimizations

- **In-place normalization**: `tensor.div_(255.0)` instead of creating new tensors
- **GPU tensor slicing**: Slices stay on GPU memory
- **Reduced CPU-GPU transfers**: Minimal data movement between devices

### 📋 Requirements

- PyTorch with CUDA support
- CUDA 11.6+ (recommended for nvjpeg stability)
- Ultralytics YOLO models

---



## <div align="center">Overview</div>

SAHI helps developers overcome real-world challenges in object detection by enabling **sliced inference** for detecting small objects in large images. It supports various popular detection models and provides easy-to-use APIs.

| Command  | Description  |
|---|---|
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)  | perform sliced/standard video/image prediction using any [ultralytics](https://github.com/ultralytics/ultralytics)/[mmdet](https://github.com/open-mmlab/mmdetection)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads)/[torchvision](https://pytorch.org/vision/stable/models.html#object-detection) model - see [CLI guide](docs/cli.md#predict-command-usage) |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage)  | perform sliced/standard prediction using any supported model and explore results in [fiftyone app](https://github.com/voxel51/fiftyone) - [learn more](docs/fiftyone.md) |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)  | automatically slice COCO annotation and image files - see [slicing utilities](docs/slicing.md) |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)  | explore multiple prediction results on your COCO dataset with [fiftyone ui](https://github.com/voxel51/fiftyone) ordered by number of misdetections |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)  | evaluate classwise COCO AP and AR for given predictions and ground truth - check [COCO utilities](docs/coco.md) |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)  | calculate and export many error analysis plots - see the [complete guide](docs/README.md) |
| [coco yolo](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolo-command-usage)  | automatically convert any COCO dataset to [ultralytics](https://github.com/ultralytics/ultralytics) format |

### Approved by the Community

[📜 List of publications that cite SAHI (currently 400+)](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

[🏆 List of competition winners that used SAHI](https://github.com/obss/sahi/discussions/688)

### Approved by AI Tools

SAHI's documentation is [indexed in Context7 MCP](https://context7.com/obss/sahi), providing AI coding assistants with up-to-date, version-specific code examples and API references. We also provide an [llms.txt](https://context7.com/obss/sahi/llms.txt) file following the emerging standard for AI-readable documentation. To integrate SAHI docs with your AI development workflow, check out the [Context7 MCP installation guide](https://github.com/upstash/context7#%EF%B8%8F-installation).

## <div align="center">🚀 GPU Quick Start</div>

### GPU Tensor-based Inference (This Fork)

```python
import torch
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction_gpu
from sahi.utils.cv import read_image_as_tensor

# Load model on GPU
detection_model = UltralyticsDetectionModel(
    model_path="yolo11n.pt",
    confidence_threshold=0.5,
    device="cuda",
    load_at_init=True,
    image_size=640,
)

# Read image directly as GPU tensor
image_tensor = read_image_as_tensor("image.jpg", device="cuda")

# Run GPU-optimized sliced prediction
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

### GPU Slicing Only

```python
from sahi.slicing import slice_image_gpu
from sahi.utils.cv import read_image_as_tensor

# Read and slice on GPU
image_tensor = read_image_as_tensor("large_image.jpg", device="cuda")
slices = slice_image_gpu(
    image=image_tensor,
    slice_height=512,
    slice_width=512,
    device="cuda",
)

# All slices are GPU tensors
for i, slice_img in enumerate(slices.images):
    print(f"Slice {i}: {slice_img.shape}, device: {slice_img.device}")
```

## <div align="center">Installation</div>

### Basic Installation

```bash
pip install sahi

```

<details closed>
<summary>
<big><b>Detailed Installation (Click to open)</b></big>
</summary>

- Install your desired version of pytorch and torchvision:

```console
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

(torch 2.1.2 is required for mmdet support):

```console
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

- Install your desired detection framework (ultralytics):

```console
pip install ultralytics>=8.3.161
```

- Install your desired detection framework (huggingface):

```console
pip install transformers>=4.49.0 timm
```

- Install your desired detection framework (yolov5):

```console
pip install yolov5==7.0.14 sahi==0.11.21
```

- Install your desired detection framework (mmdet):

```console
pip install mim
mim install mmdet==3.3.0
```

- Install your desired detection framework (roboflow):

```console
pip install inference>=0.50.3 rfdetr>=1.1.0
```

</details>

## <div align="center">Quick Start</div>

### Tutorials

- [Introduction to SAHI](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80) - explore the [complete documentation](docs/README.md) for advanced usage

- [Official paper](https://ieeexplore.ieee.org/document/9897990) (ICIP 2022 oral)

- [Pretrained weights and ICIP 2022 paper files](https://github.com/fcakyon/small-object-detection-benchmark)

- [2025 Video Tutorial](https://www.youtube.com/watch?v=ILqMBah5ZvI) (RECOMMENDED)

- [Visualizing and Evaluating SAHI predictions with FiftyOne](https://voxel51.com/blog/how-to-detect-small-objects/)

- ['Exploring SAHI' Research Article from 'learnopencv.com'](https://learnopencv.com/slicing-aided-hyper-inference/)

- [Slicing Aided Hyper Inference Explained by Encord](https://encord.com/blog/slicing-aided-hyper-inference-explained/)

- ['VIDEO TUTORIAL: Slicing Aided Hyper Inference for Small Object Detection - SAHI'](https://www.youtube.com/watch?v=UuOJKxn-M8&t=270s)

- [Video inference support is live](https://github.com/obss/sahi/discussions/626)

- [Kaggle notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx)

- [Satellite object detection](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98)

- [Error analysis plots & evaluation](https://github.com/obss/sahi/discussions/622) (RECOMMENDED)

- [Interactive result visualization and inspection](https://github.com/obss/sahi/discussions/624) (RECOMMENDED)

- [COCO dataset conversion](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)

- [Slicing operation notebook](demo/slicing.ipynb)

- `YOLOX` + `SAHI` demo: <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="sahi-yolox"></a>

- `YOLO12` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo12"></a>

- `YOLO11-OBB` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11-obb"></a> (NEW)

- `YOLO11` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11"></a>

- `Roboflow/RF-DETR` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="roboflow"></a> (NEW)

- `RT-DETR v2` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetrv2"></a> (NEW)

- `RT-DETR` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetr"></a>

- `HuggingFace` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-huggingface"></a>

- `YOLOv5` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolov5"></a>

- `MMDetection` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-mmdetection"></a>

- `TorchVision` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-torchvision"></a>

<a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a>

### Framework Agnostic Sliced/Standard Prediction

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

Find detailed info on using `sahi predict` command in the [CLI documentation](docs/cli.md#predict-command-usage) and explore the [prediction API](docs/predict.md) for advanced usage.

Find detailed info on video inference at [video inference tutorial](https://github.com/obss/sahi/discussions/626).

### Error Analysis Plots & Evaluation

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

Find detailed info at [Error Analysis Plots & Evaluation](https://github.com/obss/sahi/discussions/622).

### Interactive Visualization & Inspection

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6dd5f3-36dc-4267-8574-a985dd0c6578.gif">

Explore [FiftyOne integration](docs/fiftyone.md) for interactive visualization and inspection.

### Other utilities

Check the [comprehensive COCO utilities guide](docs/coco.md) for YOLO conversion, dataset slicing, subsampling, filtering, merging, and splitting operations. Learn more about the [slicing utilities](docs/slicing.md) for detailed control over image and dataset slicing parameters.

## <div align="center">Citation</div>

If you use this package in your work, please cite as:

```bibtex
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}
```

```bibtex
@software{obss2021sahi,
  author       = {Akyon, Fatih Cagatay and Cengiz, Cemil and Altinuc, Sinan Onur and Cavusoglu, Devrim and Sahin, Kadir and Eryuksel, Ogulcan},
  title        = {{SAHI: A lightweight vision library for performing large scale object detection and instance segmentation}},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5718950},
  url          = {https://doi.org/10.5281/zenodo.5718950}
}
```

## <div align="center">Contributing</div>

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!

<p align="center">
    <a href="https://github.com/obss/sahi/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=obss/sahi" />
    </a>
</p>
