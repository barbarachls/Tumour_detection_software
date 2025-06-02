import os
import shutil

# Get the dataset directories that already exist
train_pos_directory = "dataset/png_out/training/pos"
test_pos_directory = "dataset/png_out/testing/pos"
val_pos_directory = "dataset/png_out/validation/pos"
train_neg_directory = "dataset/png_out/training/neg"
test_neg_directory = "dataset/png_out/testing/neg"
val_neg_directory = "dataset/png_out/validation/neg"

# Path for the new directories
train_pos_r_directory = "reduced_dataset/png_out/training/pos"
test_pos_r_directory = "reduced_dataset/png_out/testing/pos"
val_pos_r_directory = "reduced_dataset/png_out/validation/pos"
train_neg_r_directory = "reduced_dataset/png_out/training/neg"
test_neg_r_directory = "reduced_dataset/png_out/testing/neg"
val_neg_r_directory = "reduced_dataset/png_out/validation/neg"


def reduce(source, destination):
    """
    Reduce each directory by 1/12th
    :param source: The source directory path
    :param destination: the destination directory path
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    # List all files in the source directory
    all_images = os.listdir(source)

    # Number of images to select (1 in 12)
    num_images_to_select = len(all_images) // 12

    # Randomly select images
    selected_images = all_images[::12]

    # Copy selected images to the destination directory
    for image in selected_images:
        source_path = os.path.join(source, image)
        destination_path = os.path.join(destination, image)
        shutil.copyfile(source_path, destination_path)

    print("Selected images copied to the destination directory.")


reduce(train_pos_directory, train_pos_r_directory)
reduce(val_pos_directory, val_pos_r_directory)
reduce(test_pos_directory, test_pos_r_directory)
reduce(train_neg_directory, train_neg_r_directory)
reduce(val_neg_directory, val_neg_r_directory)
reduce(test_neg_directory, test_neg_r_directory)
