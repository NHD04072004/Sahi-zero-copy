from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sahi.utils.cv import (
    Colors,
    apply_color_mask,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    read_image,
    read_image_as_tensor,
)


class TestCvUtils:
    def test_hex_to_rgb(self):
        colors = Colors()
        assert colors.hex_to_rgb("#FF3838") == (255, 56, 56)

    def test_hex_to_rgb_retrieve(self):
        colors = Colors()
        assert colors(0) == (255, 56, 56)

    @patch("sahi.utils.cv.cv2.cvtColor")
    @patch("sahi.utils.cv.cv2.imread")
    def test_read_image(self, mock_imread, mock_cvtColor):
        fake_image = "test.jpg"
        fake_image_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        fake_image_rbg_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        mock_imread.return_value = fake_image_val
        mock_cvtColor.return_value = fake_image_rbg_val

        result = read_image(fake_image)

        # mock_cv2.assert_called_once_with(fake_image)
        mock_imread.assert_called_once_with(fake_image)
        np.testing.assert_array_equal(result, fake_image_rbg_val)

    def test_apply_color_mask(self):
        image = np.array([[0, 1]], dtype=np.uint8)
        color = (255, 0, 0)

        expected_output = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)

        result = apply_color_mask(image, color)

        np.testing.assert_array_equal(result, expected_output)

    def test_get_coco_segmentation_from_bool_mask_simple(self):
        mask = np.zeros((10, 10), dtype=bool)
        result = get_coco_segmentation_from_bool_mask(mask)
        assert result == []

    def test_get_coco_segmentation_from_bool_mask_polygon(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[1:4, 1:4] = True
        mask[5:8, 5:8] = True
        result = get_coco_segmentation_from_bool_mask(mask)
        assert len(result) == 2

    def test_get_bbox_from_bool_mask(self):
        mask = np.array(
            [
                [False, False, False],
                [False, True, True],
                [False, True, True],
                [False, False, False],
            ]
        )
        expected_result = [1, 1, 2, 2]
        result = get_bbox_from_bool_mask(mask)
        assert result == expected_result

    @patch("sahi.utils.cv.requests.get")
    @patch("sahi.utils.cv.Image.open")
    def test_read_image_as_tensor(self, mock_image_open, mock_requests_get):
        # 1. Test Tensor input
        input_tensor = torch.rand(10, 10, 3)
        result = read_image_as_tensor(input_tensor)
        assert torch.equal(result, input_tensor)

        # 2. Test PIL Image input
        input_image = Image.new("RGB", (10, 10))
        result = read_image_as_tensor(input_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 10, 10)

        # 3. Test Numpy array input (HWC)
        input_numpy = np.zeros((10, 10, 3), dtype=np.uint8)
        result = read_image_as_tensor(input_numpy)
        assert result.shape == (3, 10, 10)

        # 4. Test Numpy array input (CHW)
        input_numpy_chw = np.zeros((3, 10, 10), dtype=np.uint8)
        result = read_image_as_tensor(input_numpy_chw)
        assert result.shape == (3, 10, 10)

        # 5. Test String Path
        mock_image_open.return_value = Image.new("RGB", (20, 20))
        result = read_image_as_tensor("test_path.jpg")
        assert result.shape == (3, 20, 20)

        # 6. Test URL
        class MockResponse:
            content = b"fake_content"

        mock_requests_get.return_value = MockResponse()
        result = read_image_as_tensor("http://example.com/image.jpg")
        assert result.shape == (3, 20, 20)
