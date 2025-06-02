from main import tranform_img, convert_dcm, classify

import pytest

import torch

class MockUploadedFile:
    def __init__(self, image_path):
        self.image_path = image_path

    def read(self):
        with open(self.image_path, 'rb') as f:
            return f.read()

@pytest.fixture
def mock_png_file():
    return MockUploadedFile(image_path='tests/Breast_MRI_066_pre_097-66.png')

@pytest.fixture
def mock_jpg_file():
    return MockUploadedFile(image_path='tests/Breast_MRI_066_pre_097-66.jpg')

@pytest.fixture
def mock_jpeg_file():
    return MockUploadedFile(image_path='tests/Breast_MRI_066_pre_097-66.jpeg')

def test_tranform_png(mock_png_file):
    image = [mock_png_file]

    transformed_data, _ = tranform_img(image)
    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape == (1, 1, 128, 128)

def test_tranform_jpg(mock_jpg_file):
    image = [mock_jpg_file]

    transformed_data, _ = tranform_img(image)
    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape == (1, 1, 128, 128)

def test_tranform_jpeg(mock_jpeg_file):
    image = [mock_jpeg_file]

    transformed_data, _ = tranform_img(image)
    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape == (1, 1, 128, 128)


def test_convert_dcm():
    # Test the convert_dcm function
    # Provide a sample DICOM image file as input

    sample_image = ['tests/1-097.dcm']  # Mock input DICOM file
    transformed_data, _ = convert_dcm(sample_image)

    # Assert the output format and shape
    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape == torch.Size([1, 1, 128, 128])


# Define the test cases
@pytest.mark.parametrize("model_type", ['Resnet50', 'MobileNetV3', 'DenseNet121'])
@pytest.mark.parametrize("images_type", ['dcm', 'png', 'jpg', 'jpeg'])  # Assuming you have other types of images
def test_classify(model_type, images_type, mock_png_file, mock_jpg_file, mock_jpeg_file):
    # Test the classify function with different model types and image types
    if images_type == 'dcm':
        image = ['tests/1-097.dcm']
    elif images_type == 'png':
        image = [mock_png_file]
    elif images_type == 'jpg':
        image = [mock_jpg_file]
    elif images_type == 'jpeg':
        image = [mock_jpeg_file]

    predictions, _ = classify(model_type, image, images_type)

    # Assert the output predictions
    assert isinstance(predictions, list)
    assert all(prediction in ['pos', 'neg'] for prediction in predictions)

