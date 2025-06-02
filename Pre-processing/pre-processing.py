import numpy as np
import os
import pydicom
from tqdm import tqdm
from skimage.io import imsave
import pandas as pd

data_path = 'manifest-1748870693153' # or where the original dataset  extracted from NBIA data retriever is stored
boxes_path = 'Annotation_Boxes.csv'
mapping_path = 'Breast-Cancer-MRI-filepath_filename-mapping.csv'
clinical_path = 'Clinical_and_Other_Features.csv'
target_png_dir = 'png_out'

if not os.path.exists(target_png_dir):
    os.makedirs(target_png_dir)
boxes_df = pd.read_csv(boxes_path)

# Take out of the dataset all non-fat-saturated images
mapping_df = pd.read_csv(mapping_path)
mapping_df = mapping_df[~mapping_df['original_path_and_filename'].str.contains('T1')]

# From the clinical data dataframe, delete all the rows that have a patient with two or more tumour
clinical_df = pd.read_csv(clinical_path, header=[0, 1, 2])
clinical_df = clinical_df[~clinical_df['Unnamed: 37_level_0', 'Position', 'Position (every bx positive for invasive cancer)(used during annotation)'].str.contains('R.*L|L.*R|R.*R|L.*L', regex=True)]

# only keep in the dataframe the patient with one tumour
list = clinical_df['Patient Information', 'Patient ID', 'Unnamed: 0_level_2']
mapping_df = mapping_df[mapping_df['original_path_and_filename'].str.contains('|'.join(list))]


def save_dcm_slice(dcm_fname, label, vol_idx):
    """
    Convert, save the images and label them
    :param dcm_fname: the dicom file name
    :param label: the label (pos/neg)
    :param vol_idx: the slices number
    :return: An png image
    """
    file_name = os.path.basename(row['original_path_and_filename'])

    # Create a path to save the slice .png file using the extracted filename and target label
    png_path = file_name.replace('.dcm', '-{}.png'.format(vol_idx))
    label_dir = 'pos' if label == 1 else 'neg'
    png_path = os.path.join(target_png_dir, label_dir, png_path)

    if not os.path.exists(os.path.join(target_png_dir, label_dir)):
        os.makedirs(os.path.join(target_png_dir, label_dir))

    if not os.path.exists(png_path):
        # only make the png image if it doesn't already exist

        # load DICOM file with pydicom library
        try:
            dcm = pydicom.dcmread(dcm_fname)
        except FileNotFoundError:
            # fix possible errors in filename from list
            dcm_fname_split = dcm_fname.split('/')
            dcm_fname_end = dcm_fname_split[-1]
            assert dcm_fname_end.split('-')[1][0] == '0'

            dcm_fname_end_split = dcm_fname_end.split('-')
            dcm_fname_end = '-'.join([dcm_fname_end_split[0], dcm_fname_end_split[1][1:]])

            dcm_fname_split[-1] = dcm_fname_end
            dcm_fname = '/'.join(dcm_fname_split)
            dcm = pydicom.dcmread(dcm_fname)

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

        # save final .png
        imsave(png_path, img)


# number of images for each class
N_class = 81655
# counts of images  extracted from each class
ct_negative = 0
ct_positive = 0

mapping_df.to_csv('count.txt', sep='\t', index=False)
# initialize iteration index of each patient volume
vol_idx = -1
for row_idx, row in tqdm(mapping_df.iterrows(), total=N_class * 2):
    # indices start at 1 here
    new_vol_idx = int((row['original_path_and_filename'].split('/')[1]).split('_')[-1])
    slice_idx = int(((row['original_path_and_filename'].split('/')[-1]).split('_')[-1]).replace('.dcm', ''))

    # new volume: get tumor bounding box
    if new_vol_idx != vol_idx:
        box_row = boxes_df.iloc[[new_vol_idx - 1]]
        start_slice = int(box_row['Start Slice'].iloc[0])
        end_slice = int(box_row['End Slice'].iloc[0])
        assert end_slice >= start_slice
    vol_idx = new_vol_idx

    # get DICOM filename
    dcm_fname = str(row['classic_path'])
    dcm_fname = os.path.join(data_path, dcm_fname)

    # determine slice label:
    # (1) if within 3D box, save as positive
    if slice_idx >= start_slice and slice_idx < end_slice:
        if ct_positive >= N_class:
            continue
        save_dcm_slice(dcm_fname, 1, vol_idx)
        ct_positive += 1
        with open('positive.txt', 'a') as file:
            file.write(dcm_fname)
            file.write("\n")

    # (2) if outside 3D box by >5 slices, save as negative
    elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
        if ct_negative >= N_class:
            continue
        save_dcm_slice(dcm_fname, 0, vol_idx)
        ct_negative += 1
        with open('negative.txt', 'a') as file:
            file.write(dcm_fname)
            file.write("\n")
