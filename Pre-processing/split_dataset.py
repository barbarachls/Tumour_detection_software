import os
import random
from collections import defaultdict
import shutil

# Set paths
source_directory = "png_out"
train_directory = "dataset/png_out/training"
test_directory = "dataset/png_out/testing"
val_directory = "dataset/png_out/validation"

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)


# Define ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1


def collect_image_paths(directory):
    """
    collect the images path from the directory
    :param directory: The source directory wehre the images are
    :return:  The images path in a list
    """
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Collect all image paths
all_image_paths = collect_image_paths(source_directory)

# Group images by patient ID
patient_images = defaultdict(list)
for image_path in all_image_paths:
    patient_id = os.path.basename(image_path).split('_')[2]
    patient_images[patient_id].append(image_path)

# Shuffle patient IDs
random.seed(42)  # For reproducibility
patient_ids = list(patient_images.keys())
random.shuffle(patient_ids)

# Split patient IDs into training, testing, and validation sets
num_patients = len(patient_ids)
num_train = int(num_patients * train_ratio)
num_test = int(num_patients * test_ratio)
num_val = num_patients - (num_train+num_test)

train_patients = patient_ids[:num_train]
test_patients = patient_ids[num_train:num_train + num_test]
val_patients = patient_ids[num_train + num_test:]


# Function to combine images for each set and separate positive and negative
def combine_images(patients, destination):
    """
    Copy all image by patient into the new directory and separate positive and negative
    :param patients: The list of patient in the new dataset
    :param destination: the destination directory
    :return:
    """
    for patient_id in patients:
        images = patient_images[patient_id]
        pos_dir = os.path.join(destination, "pos")
        neg_dir = os.path.join(destination, "neg")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # Putting neg first as some of the images are post MRi and therfore have a 'pos' in heir filename although they are neg
        for image_path in images:
            if "neg" in image_path:
                shutil.copy(image_path, os.path.join(neg_dir, os.path.basename(image_path)))
            elif "pos" in image_path:
                shutil.copy(image_path, os.path.join(pos_dir, os.path.basename(image_path)))


# Combine images for each set
combine_images(train_patients, train_directory)
combine_images(test_patients, test_directory)
combine_images(val_patients, val_directory)

print('done')