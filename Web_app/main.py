from io import BytesIO
from cv2 import imread
from mpmath.identification import transforms
import pydicom
import numpy as np
from skimage.io import imread
import torch
from torchvision import transforms
from torchvision.models import resnet50, mobilenet_v3_large, densenet121
from torch import nn
import streamlit as st

global but1, but2
global images
predictions_final = []
imgs = []


def uploading_files(type):
    """
    function that allows user to upload their file
    :param type: the type of image to be uploaded
    :return: The images
    """
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    files = st.file_uploader(
        "Please upload the MRI slices you want to examine",
        accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
        type=type
    )

    if files:
        st.session_state["uploaded_files"] = files

        col = st.columns(3)
        with col[1]:
            if st.button("Clear uploaded files"):
                st.session_state["file_uploader_key"] += 1
                st.rerun()

    return files


def tranform_img(image):
    """
    Cnvert the image to be able to use it in our model
    :param image: the image
    :return: the image converted
    """
    img_arr = imread(BytesIO(image[0].read()), as_gray=True)
    img_arr = img_arr.astype(np.float32) * 255. / img_arr.max()
    img_arr = img_arr.astype(np.uint8)

    # Convert to tensor (PyTorch matrix)

    data = torch.from_numpy(img_arr)
    data = data.type(torch.FloatTensor)

    # Add image channel dimension
    data = torch.unsqueeze(data, 0)  # Adding batch dimension
    data = torch.unsqueeze(data, 0)  # Adding channel dimension for grayscale image

    # Resize image
    data = transforms.Resize((128, 128))(data)

    return data, image


def convert_dcm(image):
    """
    Convert the dicom file to use it in our models
    :param image: the dicom file
    :return: teh converted image
    """
    dcm = pydicom.dcmread(image[0])
    # convert DICOM into numerical numpy array of pixel intensity values
    img = dcm.pixel_array

    # convert uint16 datatype to float, scaled properly for uint8
    img = img.astype(np.float32) * 255. / img.max()
    # convert from float -> uint8
    img = img.astype(np.uint8)
    # invert image if necessary, according to DICOM metadata
    img_type = dcm.PhotometricInterpretation
    if img_type == "MONOCHROME1":
        img = np.invert(img)

    # Convert to tensor (PyTorch matrix)
    data = torch.from_numpy(img)
    data = data.type(torch.FloatTensor)

    # Add image channel dimension
    data = torch.unsqueeze(data, 0)  # Adding batch dimension
    data = torch.unsqueeze(data, 0)  # Adding channel dimension for grayscale image

    # Resize image
    data = transforms.Resize((128, 128))(data)

    return data, img


def classify(model_type, images, images_type):
    """
    classify teh images
    :param model_type: what model to use
    :param images: the images to classify
    :param images_type: the image type
    :return: the predictions with the images
    """
    final_predictions = []
    if model_type == 'Resnet50':
        model = resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('Resnet50.pt', map_location=torch.device('cpu')))
        model.eval()
    elif model_type == 'MobileNetV3':
        model = mobilenet_v3_large()
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.load_state_dict(torch.load('MobileNetv3.pt', map_location=torch.device('cpu')))
        model.eval()
    else:
        model = densenet121()
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.load_state_dict(torch.load('DenseNet121.pt', map_location=torch.device('cpu')))
        model.eval()

    for image in images:
        if images_type == 'dcm':
            input_data, img = convert_dcm([image])
            imgs.append(img)
        else:
            input_data, img = tranform_img([image])
            imgs.append(img)

        with torch.no_grad():
            # Make prediction
            predictions = model(input_data)

            # Get predicted class
            _, predicted_class = predictions.max(1)

            # Convert predicted_class tensor to integer
            predicted_class = predicted_class.item()

        # Assuming your classes are named "neg" and "pos"
        class_names = ["neg", "pos"]

        # Get the predicted class label
        predicted_label = class_names[predicted_class]
        final_predictions.append(predicted_label)

    return final_predictions, imgs


def show_predictions(final_predictions, imgs):
    """
    Display the predictions
    :param final_predictions: the predictions of the model
    :param imgs: teh images
    """
    j = 0
    # Display images in groups of 4 on each row
    for i in range(0, len(imgs), 4):
        row_images = imgs[i:i + 4]  # Get 4 images for the row
        col1, col2, col3, col4 = st.columns(4)  # Create 4 columns for the row

        for image_path, col in zip(row_images, [col1, col2, col3, col4]):
            if final_predictions[j] == 'pos':
                caption = 'positive'
            elif final_predictions[j] == 'neg':
                caption = 'negative'
            with col:
                st.image(image_path, caption=caption, use_column_width=True)
                j = j + 1

# tile and welcome paragraph
st.title('Welcome')
st.markdown("Welcome to the automated tumour detection service.")

# The main application with teh different functionalities
images_type = st.selectbox('Select the image type', ('dcm', 'png', 'jpg', 'jpeg'), index=None, placeholder="select the type of the files")
if images_type:
    images = uploading_files(images_type)
    if images:
        model_type = st.selectbox('Select deep learning model', ('Resnet50', 'MobileNetV3', 'DenseNet121'), index=None, placeholder="select the model you want to use")
        if model_type:
            if st.button('classifiy', type="primary"):
                final_predicitons, imgs = classify(model_type, images, images_type)
                show_predictions(final_predicitons, imgs)
