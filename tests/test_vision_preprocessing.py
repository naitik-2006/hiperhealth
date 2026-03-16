import pytest
import io
import cv2
import numpy as np
from hiperhealth.vision.preprocessing import ImageProcessor, ImageValidationError
from PIL import Image, ImageFilter, ImageDraw

@pytest.fixture
def processor():
    return ImageProcessor()

@pytest.fixture
def valid_image_bytes():
    # Create a high-contrast sharp image with average grey background instead of pure white to pass exposure test
    img = Image.new('RGB', (500, 500), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    # Adding sharp lines to ensure high Laplacian variance
    for i in range(0, 500, 50):
        draw.line([(i, 0), (i, 500)], fill="black", width=2)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def blurry_image_bytes():
    # Create a sharp image then blur it heavily
    img = Image.new('RGB', (300, 300), color='red')
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# --- Tests ---

def test_validate_integrity_success(processor, valid_image_bytes):
    """Should return True for a high-quality, sharp image."""
    assert processor.validate_integrity(valid_image_bytes) is True

def test_validate_integrity_fails_on_blur(processor, blurry_image_bytes):
    """Should raise ImageValidationError for blurry images."""
    with pytest.raises(ImageValidationError, match="too blurry"):
        processor.validate_integrity(blurry_image_bytes)

def test_validate_integrity_fails_on_corrupt(processor):
    """Should raise ImageValidationError for random bytes."""
    bad_data = b"not_an_image_file_at_all"
    with pytest.raises(ImageValidationError):
        processor.validate_integrity(bad_data)

def test_standardize_logic(processor, valid_image_bytes):
    """Should output 224x224 RGB regardless of input size."""
    standardized_img = processor.standardize(valid_image_bytes)
    
    assert isinstance(standardized_img, Image.Image)
    assert standardized_img.size == (224, 224)
    assert standardized_img.mode == 'RGB'

def test_standardize_converts_rgba(processor):
    """Ensure transparency (RGBA) is flattened to RGB."""
    rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    rgba_img.save(img_byte_arr, format='PNG')
    
    standardized_img = processor.standardize(img_byte_arr.getvalue())
    assert standardized_img.mode == 'RGB'

@pytest.fixture
def underexposed_image_bytes():
    img = Image.new('RGB', (100, 100), color=(10, 10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def overexposed_image_bytes():
    img = Image.new('RGB', (100, 100), color=(250, 250, 250))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_validate_integrity_fails_on_underexposed(processor, underexposed_image_bytes):
    with pytest.raises(ImageValidationError, match="underexposed"):
        processor.validate_integrity(underexposed_image_bytes)

def test_validate_integrity_fails_on_overexposed(processor, overexposed_image_bytes):
    with pytest.raises(ImageValidationError, match="overexposed"):
        processor.validate_integrity(overexposed_image_bytes)