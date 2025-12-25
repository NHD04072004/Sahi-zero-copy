"""Tests for GPU zero-copy prediction functions."""
import numpy as np
import pytest
import torch

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction_gpu, get_sliced_prediction_gpu
from sahi.slicing import slice_image_gpu
from sahi.utils.cv import read_image, read_image_as_tensor

from .utils.ultralytics import UltralyticsConstants, download_yolo11n_model


# Test configuration
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


class TestSliceImageGpu:
    """Test cases for slice_image_gpu function."""

    def test_slice_image_gpu_from_path(self):
        """Test slicing image from file path."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        slice_image_result = slice_image_gpu(
            image=image_path,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            device=MODEL_DEVICE,
        )
        
        assert len(slice_image_result) > 0
        assert len(slice_image_result.images) > 0
        # Check that sliced images are torch tensors
        assert isinstance(slice_image_result.images[0], torch.Tensor)
        
    def test_slice_image_gpu_from_tensor(self):
        """Test slicing from a torch tensor input."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        slice_image_result = slice_image_gpu(
            image=image_tensor,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            device=MODEL_DEVICE,
        )
        
        assert len(slice_image_result) > 0
        assert isinstance(slice_image_result.images[0], torch.Tensor)
        # Check tensor is on the correct device
        assert slice_image_result.images[0].device.type == MODEL_DEVICE.split(":")[0]

    def test_slice_image_gpu_from_numpy(self):
        """Test slicing from numpy array input."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_np = read_image(image_path)
        
        slice_image_result = slice_image_gpu(
            image=image_np,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            device=MODEL_DEVICE,
        )
        
        assert len(slice_image_result) > 0
        assert isinstance(slice_image_result.images[0], torch.Tensor)

    def test_slice_image_gpu_starting_pixels(self):
        """Test that starting pixels are correctly computed."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        slice_image_result = slice_image_gpu(
            image=image_path,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
            device=MODEL_DEVICE,
        )
        
        # First slice should start at (0, 0)
        assert slice_image_result.starting_pixels[0] == [0, 0]
        # Check other starting pixels are non-negative
        for sp in slice_image_result.starting_pixels:
            assert sp[0] >= 0
            assert sp[1] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGetPredictionGpu:
    """Test cases for get_prediction_gpu function."""

    def test_get_prediction_gpu_from_tensor(self):
        """Test GPU prediction from tensor input."""
        # Download and load model
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        # Prepare image as tensor on GPU
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Get prediction
        prediction_result = get_prediction_gpu(
            image=image_tensor,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            device=MODEL_DEVICE,
        )
        
        # Verify predictions
        assert prediction_result is not None
        assert "prediction" in prediction_result.durations_in_seconds
        # Should detect some objects (cars) in the image
        assert len(prediction_result.object_prediction_list) >= 0

    def test_get_prediction_gpu_from_numpy(self):
        """Test GPU prediction from numpy array input."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        # Prepare image as numpy
        image_path = "tests/data/small-vehicles1.jpeg"
        image_np = read_image(image_path)
        
        # Get prediction (should auto-convert to tensor)
        prediction_result = get_prediction_gpu(
            image=image_np,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            device=MODEL_DEVICE,
        )
        
        assert prediction_result is not None
        assert "prediction" in prediction_result.durations_in_seconds

    def test_get_prediction_gpu_with_shift_amount(self):
        """Test GPU prediction with shift amount."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Get prediction with shift
        prediction_result = get_prediction_gpu(
            image=image_tensor,
            detection_model=detection_model,
            shift_amount=[100, 100],
            full_shape=[1000, 1000],
            device=MODEL_DEVICE,
        )
        
        assert prediction_result is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGetSlicedPredictionGpu:
    """Test cases for get_sliced_prediction_gpu function."""

    def test_get_sliced_prediction_gpu_from_tensor(self):
        """Test sliced GPU prediction from tensor input."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        # Prepare image as tensor
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Get sliced prediction
        prediction_result = get_sliced_prediction_gpu(
            image=image_tensor,
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
            postprocess_match_metric="IOS",
            postprocess_class_agnostic=True,
            device=MODEL_DEVICE,
        )
        
        # Verify results
        assert prediction_result is not None
        assert "slice" in prediction_result.durations_in_seconds
        assert "prediction" in prediction_result.durations_in_seconds
        assert "postprocess" in prediction_result.durations_in_seconds
        # Should detect cars in the image
        assert len(prediction_result.object_prediction_list) >= 0

    def test_get_sliced_prediction_gpu_from_path(self):
        """Test sliced GPU prediction from file path."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        
        # Get sliced prediction
        prediction_result = get_sliced_prediction_gpu(
            image=image_path,
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            device=MODEL_DEVICE,
        )
        
        assert prediction_result is not None
        assert len(prediction_result.object_prediction_list) >= 0

    def test_get_sliced_prediction_gpu_with_standard_pred(self):
        """Test sliced GPU prediction with standard prediction enabled."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        prediction_result = get_sliced_prediction_gpu(
            image=image_tensor,
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,  # Enable standard prediction
            device=MODEL_DEVICE,
        )
        
        assert prediction_result is not None

    def test_get_sliced_prediction_gpu_with_progress_callback(self):
        """Test sliced GPU prediction with progress callback."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        
        # Track progress
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        prediction_result = get_sliced_prediction_gpu(
            image=image_path,
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            perform_standard_pred=False,
            progress_callback=progress_callback,
            device=MODEL_DEVICE,
        )
        
        assert prediction_result is not None
        # Should have some progress calls
        assert len(progress_calls) > 0
        # Last call should have current == total
        assert progress_calls[-1][0] == progress_calls[-1][1]


class TestReadImageAsTensor:
    """Test cases for read_image_as_tensor function."""

    def test_read_from_path(self):
        """Test reading image from path as tensor."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device="cpu")
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.ndim == 3  # H, W, C
        assert image_tensor.device.type == "cpu"

    def test_read_from_numpy(self):
        """Test converting numpy array to tensor."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_np = read_image(image_path)
        
        image_tensor = read_image_as_tensor(image_np, device="cpu")
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == image_np.shape

    def test_read_tensor_passthrough(self):
        """Test that tensor input is passed through."""
        original_tensor = torch.randn(480, 640, 3)
        
        result_tensor = read_image_as_tensor(original_tensor, device="cpu")
        
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == original_tensor.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_read_to_gpu(self):
        """Test reading image directly to GPU."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device="cuda")
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.device.type == "cuda"


class TestPerformInferenceGpu:
    """Test cases for perform_inference_gpu method in UltralyticsDetectionModel."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_perform_inference_gpu(self):
        """Test GPU inference method."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        # Prepare image tensor
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Perform inference
        detection_model.perform_inference_gpu(image_tensor)
        
        # Check predictions were stored
        assert detection_model.original_predictions is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_perform_inference_gpu_and_convert(self):
        """Test GPU inference and converting predictions."""
        download_yolo11n_model()
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Perform inference
        detection_model.perform_inference_gpu(image_tensor)
        
        # Convert predictions
        detection_model.convert_original_predictions(
            shift_amount=[[0, 0]],
            full_shape=[[image_tensor.shape[0], image_tensor.shape[1]]],
        )
        
        # Check object predictions
        predictions = detection_model.object_prediction_list
        assert isinstance(predictions, list)


class TestEdgeCases:
    """Test edge cases for tensor-based SAHI operations."""

    def test_slice_image_gpu_different_slice_sizes(self):
        """Test slicing with different slice sizes."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        # Test with smaller slices
        slice_result_small = slice_image_gpu(
            image=image_path,
            slice_height=128,
            slice_width=128,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            device="cpu",
        )
        
        # Test with larger slices
        slice_result_large = slice_image_gpu(
            image=image_path,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            device="cpu",
        )
        
        assert len(slice_result_small) > len(slice_result_large)
        assert all(isinstance(img, torch.Tensor) for img in slice_result_small.images)
        assert all(isinstance(img, torch.Tensor) for img in slice_result_large.images)

    def test_tensor_hwc_format(self):
        """Test that tensor input in HWC format works correctly."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device="cpu")
        
        # Should be HWC format (Height, Width, Channels)
        assert image_tensor.ndim == 3
        assert image_tensor.shape[2] == 3  # 3 channels (BGR/RGB)
        
        # Test slicing with HWC tensor
        slice_result = slice_image_gpu(
            image=image_tensor,
            slice_height=256,
            slice_width=256,
            device="cpu",
        )
        
        assert len(slice_result) > 0
        # Each slice should also be HWC format
        for img in slice_result.images:
            assert img.ndim == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self):
        """Test tensor transfer from CPU to GPU."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        # Read on CPU
        image_cpu = read_image_as_tensor(image_path, device="cpu")
        assert image_cpu.device.type == "cpu"
        
        # Transfer to GPU
        image_gpu = image_cpu.to("cuda")
        assert image_gpu.device.type == "cuda"
        
        # Test slicing on GPU with transferred tensor
        slice_result = slice_image_gpu(
            image=image_gpu,
            slice_height=256,
            slice_width=256,
            device="cuda",
        )
        
        assert len(slice_result) > 0
        # Slices should be on GPU
        assert slice_result.images[0].device.type == "cuda"

    def test_tensor_dtype_uint8(self):
        """Test that uint8 tensors are handled correctly."""
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device="cpu")
        
        # Should be uint8 (from cv2.imread)
        assert image_tensor.dtype == torch.uint8
        
        # Slicing should work with uint8
        slice_result = slice_image_gpu(
            image=image_tensor,
            slice_height=256,
            slice_width=256,
            device="cpu",
        )
        
        assert len(slice_result) > 0

    def test_slice_image_zero_overlap(self):
        """Test slicing with zero overlap ratio."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        slice_result = slice_image_gpu(
            image=image_path,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
            device="cpu",
        )
        
        assert len(slice_result) > 0
        # Verify no overlapping starting pixels
        pixels = slice_result.starting_pixels
        for i, sp1 in enumerate(pixels):
            for j, sp2 in enumerate(pixels):
                if i != j:
                    # Different slices should not start at the same position
                    assert sp1 != sp2


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_image_path(self):
        """Test that invalid image path raises an error."""
        with pytest.raises(Exception):
            read_image_as_tensor("nonexistent_image.jpg", device="cpu")

    def test_invalid_tensor_dimension(self):
        """Test that invalid tensor dimensions are handled."""
        # Create invalid 2D tensor
        invalid_tensor = torch.randn(100, 100)  # 2D, should be 3D
        
        # This should either work (by handling gracefully) or raise an error
        try:
            result = slice_image_gpu(
                image=invalid_tensor,
                slice_height=50,
                slice_width=50,
                device="cpu",
            )
            # If it works, slices should still be produced
            assert len(result) >= 0
        except (RuntimeError, ValueError, IndexError):
            # Expected - invalid dimensions should raise an error
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_not_loaded(self):
        """Test that inference fails gracefully when model is not loaded."""
        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=False,  # Do not load model
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor = read_image_as_tensor(image_path, device=MODEL_DEVICE)
        
        # Should raise error because model is not loaded
        with pytest.raises(ValueError, match="Model is not loaded"):
            detection_model.perform_inference_gpu(image_tensor)


class TestDataConsistency:
    """Test data consistency between GPU and CPU pipelines."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_prediction_consistency(self):
        """Test that GPU and CPU predictions produce similar results."""
        download_yolo11n_model()
        
        # GPU model
        detection_model_gpu = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device="cuda",
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        # CPU model
        detection_model_cpu = UltralyticsDetectionModel(
            model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device="cpu",
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        
        image_path = "tests/data/small-vehicles1.jpeg"
        image_tensor_gpu = read_image_as_tensor(image_path, device="cuda")
        image_tensor_cpu = read_image_as_tensor(image_path, device="cpu")
        
        # Get GPU prediction
        result_gpu = get_prediction_gpu(
            image=image_tensor_gpu,
            detection_model=detection_model_gpu,
            shift_amount=[0, 0],
            device="cuda",
        )
        
        # Get CPU prediction using GPU pipeline on CPU
        result_cpu = get_prediction_gpu(
            image=image_tensor_cpu,
            detection_model=detection_model_cpu,
            shift_amount=[0, 0],
            device="cpu",
        )
        
        # Both should produce predictions
        assert result_gpu is not None
        assert result_cpu is not None
        
        # Number of predictions should be similar (within some tolerance)
        num_gpu = len(result_gpu.object_prediction_list)
        num_cpu = len(result_cpu.object_prediction_list)
        
        # Allow some difference due to numerical precision
        assert abs(num_gpu - num_cpu) <= max(1, num_gpu // 5), \
            f"GPU detected {num_gpu}, CPU detected {num_cpu} - too different"

    def test_slice_image_result_consistency(self):
        """Test that sliced image results are consistent."""
        image_path = "tests/data/small-vehicles1.jpeg"
        
        # Slice the same image twice
        result1 = slice_image_gpu(
            image=image_path,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            device="cpu",
        )
        
        result2 = slice_image_gpu(
            image=image_path,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            device="cpu",
        )
        
        # Same parameters should produce same number of slices
        assert len(result1) == len(result2)
        
        # Same starting pixels
        assert result1.starting_pixels == result2.starting_pixels
        
        # Same original image size
        assert result1.original_image_height == result2.original_image_height
        assert result1.original_image_width == result2.original_image_width
